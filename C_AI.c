#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <getopt.h>

#include "common.h"
#include "board.h"
#include "bitboard.h"
#include "AI.h"
#include "threadpool.h"

struct game_struct {
    AI_instance_t **ais;
    int games_to_play, max_moves;
    int (*do_a_move)(board_t *);
    char *fen;
    int game_id;
    struct uci *engine;
};

int nr_threads = 2, nr_jobs = 2, i, best, iteration;
struct game_struct *games = NULL;
struct job *jobs = NULL;

char *ai_file = NULL;
int games_to_play = 100;
int max_iterations = 100;
int brain_size = 1;
int selection_function = 1;
int nr_selections = 1;
int games_pr_iteration = 50;
int nr_ports = 256;
int max_moves = 20;
int ai_vs_ai = 0;
AI_instance_t **ais;
AI_instance_t ***ais_2d;
char uci_engine[256] = "";
int nr_islands = 1;

void print_ai_stats(int tid, AI_instance_t *ai, int ite, int rndwins)
{
    printf("thread %d: iteration %d\n", tid, ite);
    printf("thread %d: aiw nr wins: %d\n", tid, ai->nr_wins);
    printf("thread %d: random wins: %d\n", tid, rndwins);
    printf("thread %d: memory attributes:\n", tid);
    printf("    P and R\n");
    printf("thread %d: generation: %d\n", tid, ai->generation);
}
pthread_mutex_t lock;

void play_chess(void *arg)
{
    struct game_struct *game = (struct game_struct *) arg;
    int nr_games, moves, ret = -1;
    AI_instance_t *ai;
    board_t *board;

    ai = game->ais[game->game_id];
    // printf("game_id: %d\n", game->game_id);

    for (nr_games = 0; nr_games < games_pr_iteration; nr_games++) {
        //  printf("__________________NEW GAME_________________\n");
        board = new_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
        if (board == NULL) {
            fprintf(stderr, "ERROR: BOARD = NULL!!!\n");
            exit(1);
        }

        if (uci_engine[0])
            uci_new_game(game->engine, "fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");

        for (moves = 0; moves < max_moves; moves++) {
            //printf("AI move %d\n", game->game_id);
            ret = do_best_move(ai, board, game->engine);
            //ret = do_random_move(board, game->engine);
            //printf("AI move done %d\n", game->game_id);
            if (ret == 0) {
                break;
            } else if (ret == -1) {
                punish(ai);
                //printf("AI lost\n");
                //                   print_board(board->board);
                break;
            }


            if (uci_engine[0]) {
                if (!random_int_r(0, 2)) {
                    ret = do_random_move(board, game->engine);
                } else {
                    ret = do_uci_move(board, game->engine);
                }
            }

            if (ret == 0) {
                break;
            } else if (ret == -1) {
                //printf("AI won\n");
                reward(ai);
                break;

            }
            /*
               print_board(board->board);
               getchar();
             */

        }


        //nobody won, either because of stalemate or max number of moves made
        if (ret == 0 || moves == max_moves) {
            draw(ai, board);
            //printf("DRAW\n");
        }
        free_board(board);

    }
    // printf("done game_id: %d\n", game->game_id);
}

int get_best_ai(AI_instance_t **ais, int n, int lim)
{
    int i, best = lim ? 0 : 1;
    float best_val = -1000000.0;
    for (i = 0; i < n; i++) {
        if (get_score(ais[i]) > best_val && lim != i && games_to_play <= ais[i]->nr_games_played) {
            best = i;
            best_val = get_score(ais[i]);
        }
    }

    return best;
}

//used to catch SIGINT and interrupt the training to write the AI to file 
void sighandler(int sig)
{
    char buffer[512], file[128];
    int n = 1, i, best;
    struct timeval tv;
    struct tm *tm;

    gettimeofday(&tv, NULL);
    tm = localtime(&tv.tv_sec);
    strftime(file, 64, "ai_save-%d%m%y-%H%M", tm);

    fprintf(stderr, "Are you sure you want to quit? (Y/n): [default: n] ");
    if (fgets(buffer, sizeof (buffer), stdin) == NULL)
        return;
    if (tolower(buffer[0]) == 'c') //exit without any further questions
        exit(0);

    if (tolower(buffer[0]) != 'y')
        return;


    fprintf(stderr, "Save N best AIs to file? (Y/n): [default: y] ");
    if (fgets(buffer, sizeof (buffer), stdin) == NULL)
        exit(0);
    if (tolower(buffer[0]) == 'n')
        n = 0;

    if (n) {
        fprintf(stderr, "N (0-%d): [default: %d] ", nr_jobs, n);
        if (fgets(buffer, sizeof (buffer), stdin) == NULL)
            return;
        buffer[127] = 0;
        if (buffer[0] != '\n')
            n = atoi(buffer);
        if (n > nr_jobs) {
            fprintf(stderr, "N > %d, setting N to %d\n", nr_jobs, nr_jobs);
            n = nr_jobs;
        }

        fprintf(stderr, "File prefix (<= 64 chars): [default: %s] ", file);
        if (fgets(buffer, 64, stdin) == NULL)
            return;
        if (buffer[0] != '\n') {
            strncpy(file, buffer, 64);
            if (file[strlen(file) - 1] == '\n')
                file[strlen(file) - 1] = 0;
            file[64] = 0;
        }
    }

    for (i = 0; i < n; i++) {
        best = get_best_ai(ais, nr_jobs, -1);
        sprintf(buffer, "%s-%d.aidump", file, i + 1);
        fprintf(stderr, "saving ai to file '%s'\n", buffer);
        dump_ai(buffer, ais[best]);
        clear_score(ais[best]);
    }
    printf("ai saved\n");

    printf("shutting down threadpool\n");
    shutdown_threadpool(1);
    for (i = 0; i < nr_jobs * nr_islands; i++) {
        fprintf(stderr, "closing engine %d\n", i);
        uci_close(games[i].engine);
    }
    exit(0);
}

void usage(char **argv, struct option *options)
{
    int i;
    printf("USAGE: %s <options>\n", argv[0]);
    printf("Available options:\n");
    for (i = 0; options[i].name; i++)
        printf("    -%c, --%s %s\n", options[i].val, options[i].name,
            options[i].has_arg == required_argument ? "<argument>" : "");
}

void parse_arguments(int argc, char **argv)
{
    int c;
    int option_index = 0;
    static struct option long_options[] = {
        {"threads", required_argument, NULL, 't'},
        {"jobs", required_argument, NULL, 'j'},
        {"file", required_argument, NULL, 'f'},
        {"uci-enine", required_argument, NULL, 'u'},
        {"games-to-play", required_argument, NULL, 'g'},
        {"iterations", required_argument, NULL, 'i'},
        {"games-pr-iteration", required_argument, NULL, 'n'},
        {"ports", required_argument, NULL, 'p'},
        {"moves", required_argument, NULL, 'm'},
        {"training-mode", required_argument, NULL, 'a'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    while ((c = getopt_long(argc, argv, "t:j:f:g:i:n:p:m:a:u:h", long_options,
            &option_index)) != -1)
        switch (c) {
            case 't':
                nr_threads = atoi(optarg);
                break;
            case 'j':
                nr_jobs = atoi(optarg);
                break;
            case 'f':
                ai_file = optarg;
                break;
            case 'g':
                games_to_play = atoi(optarg);
                break;
            case 'i':
                max_iterations = atoi(optarg);
                break;
            case 'n':
                games_pr_iteration = atoi(optarg);
                break;
            case 'p':
                nr_ports = atoi(optarg);
                break;
            case 'm':
                max_moves = atoi(optarg);
                break;
            case 'a':
                ai_vs_ai = atoi(optarg);
                break;
            case 'u':
                strncpy(uci_engine, optarg, sizeof (uci_engine));
                break;
            case 'h':
            default:
                usage(argv, long_options);
                exit(0);
        }

    if (nr_threads == 0 || nr_jobs == 0) {
        printf("threads or jobs cannot be 0\n");
        usage(argv, long_options);
        exit(1);
    }
}

//prints the brain in a graph format, tmp == 1 means the brain is written to brain.dot
int print_brain(AI_instance_t *a1, int tmp)
{
    int i, j;

    char filename[0x100];
    snprintf(filename, sizeof (filename), tmp ? "brain.dot" : "brain_%f_%d_%d.dot", get_score(a1), max_moves, a1->nr_ports);
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    /* print some text */
    const char *text = "brain";
    fprintf(f, "digraph %s{\n", text);


    for (i = a1->low_port; i <= a1->high_port; i++) {

        for (j = 0; j < a1->nr_synapsis; j++) {
            if (((TestBit(a1->used_port, i) || a1->output_tag[i])) && TestBit(a1->brain[i], j)) {
                piece_t piece = 1 << ((j % (8 * 16)) % 16);

                fprintf(f, "p_%d -> p_%d\n", i, j);
                if (j >= a1->nr_ports + a1->board_size / 2)
                    fprintf(f, "p_%d[label=\"x:%d y:%d piece:%s\"]\n", j, j / (8 * 16), (j % (8 * 16)) / 16, piece_to_str(piece));
                else if (j >= a1->nr_ports)
                    fprintf(f, "p_%d[label=\"x:%d y:%d piece:%s(M)\"]\n", j, j / (8 * 16), (j % (8 * 16)) / 16, piece_to_str(piece));


                if (a1->port_type[i] == 1)
                    fprintf(f, "p_%d[color=red,style=filled,shape=octagon]\n", i);
                else if (a1->port_type[i] == 2)
                    fprintf(f, "p_%d[color=green,style=filled,shape=doublecircle]\n", i);
                else if (a1->port_type[i] == 3)
                    fprintf(f, "p_%d[color=chocolate2,style=filled,shape=star]\n", i);
                else if (a1->port_type[i] == 4)
                    fprintf(f, "p_%d[color=deeppink3,style=filled,shape=house]\n", i);
                if (a1->output_tag[i])
                    fprintf(f, "p_%d[color=gold,style=filled]\n", i);
            }
        }
    }
    fprintf(f, "{ rank = sink;"
            "Legend   [shape=none, margin=0, label=<"
            "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">"
            "<TR>"
            " <TD COLSPAN=\"2\"><B>Legend</B></TD>"
            "</TR>"

            "<TR>"
            "<TD>AI attributes</TD>"
            "<TD>score: %f, max_moves: %d, ports: %d</TD>"
            "</TR>"

            "<TR>"
            "<TD>Output</TD>"
            "<TD BGCOLOR=\"gold\"></TD>"
            "</TR>"



            "<TR>"
            "<TD>AND</TD>"
            "<TD BGCOLOR=\"deeppink3\"></TD>"
            "</TR>"

            "<TR>"
            "<TD>OR</TD>"
            "<TD BGCOLOR=\"chocolate2\"></TD>"
            "</TR>"

            "<TR>"
            "<TD>NAND</TD>"
            "<TD BGCOLOR=\"red\"></TD>"
            "</TR>"

            "<TR>"
            "<TD>NOR</TD>"
            "<TD BGCOLOR=\"green\"></TD>"
            "</TR>"


            "</TABLE>"
            ">];"
            "}", get_score(a1), max_moves, a1->nr_ports);
    fprintf(f, "}");

    fclose(f);

    printf("brain written to %s\n", filename);
    return 1;
}

int main(int argc, char *argv[])
{
    int j;
    int k;
    parse_arguments(argc, argv);

    srand(100);
    init_threadpool(nr_threads);
    init_magicmoves();
    signal(SIGINT, sighandler);

    //allocate structs for AI, thread jobs and game data
    ais_2d = (AI_instance_t***) malloc_2d(nr_jobs, nr_islands, sizeof (struct AI_instance*));
    ais = malloc(nr_islands * nr_jobs * sizeof (struct AI_instance*));
    jobs = malloc(nr_islands * nr_jobs * sizeof (struct job));
    games = malloc(nr_islands * nr_jobs * sizeof (struct game_struct));

    // initialize jobs to the threadpool
    for (i = 0, k = 0; k < nr_jobs; k++) {
        for (j = 0; j < nr_islands; j++, i++) {
            if (ai_file) {
                ais_2d[j][k] = load_ai(ai_file);
                clear_score(ais_2d[j][k]);

                ais[i] = ais_2d[j][k];
            } else {
                ais[i] = ai_new(nr_ports);
                ais_2d[j][k] = ais[i];
            }
            if (ais_2d[j][k] == NULL) {
                perror("ai creation");
                exit(1);
            }

            games[i].ais = ais;
            games[i].fen = DEFAULT_FEN;
            games[i].game_id = i;

            //setup the UCI engine
            if (uci_engine[0]) {
                games[i].engine = uci_init(uci_engine, "fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", BLACK);
                if (games[i].engine == NULL) {
                    printf("failed to initialize uci engine\n");
                    exit(1);
                }
            }

            jobs[i].data = games + i;
            jobs[i].task = play_chess;
        }
    }
    // post jobs to the threadpool
    for (i = 0; i < nr_islands * nr_jobs; i++)
        put_new_job(jobs + i);

    iteration = 0;
    while (iteration < max_iterations) {
        printf("iteration %d\n", ++iteration);

        /* wait for jobs to finish */
        while (get_jobs_left() > 0 || get_jobs_in_progess() > 0)
            usleep(1000 * 1); // sleep 1 ms
        printf("jobs done\n");

        int nr_mutated = 0;
        best = get_best_ai(ais, nr_jobs, -1);
        if (!(iteration % 100)) {
            printf("print tmp brain\n");
            print_brain(ais[best], 1);
        }
        for (k = 0, i = 0; k < nr_islands; k++) {
            best = get_best_ai(ais_2d[k], nr_jobs, -1);
            printf("nr played %d\n", ais_2d[k][best]->nr_games_played);

            for (j = 0; j < nr_jobs; put_new_job(jobs + i), j++, i++) {
                if (games_to_play > ais_2d[k][best]->nr_games_played) {
                    printf("need to play more before mutating\n");
                    continue;
                }
                if (j == best) {
                    //                  printf("posting job %d\n", i);
                    continue;
                }
                if (get_score(ais_2d[k][j]) < get_score(ais_2d[k][best])) {
                    mutate(ais_2d[k][j], ais_2d[k][best], 0, !nr_mutated);
                    nr_mutated++;
                }
                //printf("posting job %d\n", i);
            }
        }
        printf("Percent mutated: %f\n", ((float) nr_mutated) / (nr_jobs - 1));

    }
    while (get_jobs_left() > 0 || get_jobs_in_progess() > 0)
        usleep(1000 * 1); // sleep 1 ms

    printf("jobs done\n");
    shutdown_threadpool(1);

    best = get_best_ai(ais, nr_jobs, -1);

    for (i = 0; i < nr_jobs * nr_islands; i++)
        uci_close(games[i].engine);

    print_brain(ais[best], 0);

    dump_ai("ai.aidump", ais[best]);
    clear_score(ais[best]);

    return 0;
}
