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
#include "AI.h"
#include "threadpool.h"

struct game_struct {
    AI_instance_t **ais;
    int games_to_play, max_moves;
    int (*do_a_move)(board_t *);
    char *fen;
    int game_id;
};

int nr_threads = 2, nr_jobs = 2, i, best, iteration;
struct game_struct *games = NULL;
struct job *jobs = NULL;

char *ai_file = NULL;
int mutation_rate = 1000;
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

void print_ai_stats(int tid, AI_instance_t *ai, int ite, int rndwins) {
    printf("thread %d: iteration %d\n", tid, ite);
    printf("thread %d: aiw nr wins: %d\n", tid, ai->nr_wins);
    printf("thread %d: random wins: %d\n", tid, rndwins);
    printf("thread %d: memory attributes:\n", tid);
    printf("    P and R\n");
    printf("thread %d: generation: %d\n", tid, ai->generation);
}

void play_chess(void *arg) {
    struct game_struct *game = (struct game_struct *) arg;
    int nr_games, moves, ret = -1;
    AI_instance_t *ai;

    board_t *board;

    ai = game->ais[game->game_id];
    //printf("game_id: %d\n", game->game_id);
    //printf("starting game %d\n", game->game_id);
    //    printf("max_moves: %d\n", max_moves);
    for (nr_games = 0; nr_games < games_pr_iteration; nr_games++) {
        board = new_board(game->fen);
        if (board == NULL) {
            fprintf(stderr, "BOARD = NULL!!!\n");
            exit(1);
        }
        //board_t *board = new_board("rnbqkbnr/qqqqqqqq/8/8/8/8/qqqqqqqq/qqqqKqqq w - - 0 1");

        for (moves = 0; moves < max_moves; moves++) {
            ret = do_best_move(ai, board);
            //                              print_board(board->board);

            if (ret == 0) {
                break;
            } else if (ret == -1) {

                punish(ai);
                //printf("AI lost\n");
                //            print_board(board->board);
                break;
            }

            ret = do_random_move(board);
            if (ret == 0) {
                break;
            } else if (ret == -1) {
                //printf("AI won\n");
                //                                       print_board(board->board);
                reward(ai);
                break;
            }
        }
        if (ret == 0 || moves == max_moves)
            draw(ai, board);
        //  if (ret >= 0){
        //          small_reward(ai,score_board(board->board));            
        //  }
        free_board(board);
    }
}

void play_chess_aivsai(void *arg) {
    struct game_struct *game = (struct game_struct *) arg;
    int nr_games, moves, ret = -1;
    AI_instance_t *ai;
    AI_instance_t *ai_b;

    board_t *board;
    if (game->game_id >= nr_jobs / 2)
        return;
    ai = game->ais[game->game_id];
    ai_b = game->ais[game->game_id + nr_jobs / 2];

    //printf("game_id: %d\n", game->game_id);
    //printf("starting game %d\n", game->game_id);

    for (nr_games = 0; nr_games < games_pr_iteration; nr_games++) {
        board = new_board(game->fen);
        //board_t *board = new_board("rnbqkbnr/qqqqqqqq/8/8/8/8/qqqqqqqq/qqqqKqqq w - - 0 1");

        for (moves = 0; moves < max_moves; moves++) {
            ret = do_best_move(ai, board);

            //          ret = do_best_move(ai, board);
            //                    print_board(board->board);

            if (ret == 0) {
                break;
            } else if (ret == -1) {
                reward(ai_b);

                punish(ai);
                //printf("AI lost\n");
                //            print_board(board->board);
                break;
            }
            ret = do_best_move(ai_b, board);


            //            ret = do_best_move(ai_b, board);
            if (ret == 0) {
                break;
            } else if (ret == -1) {
                //printf("AI won\n");
                //                             print_board(board->board);
                reward(ai);
                punish(ai_b);

                break;
            }
        }
        if (ret == 0 || moves == max_moves) {
            draw(ai, board);
            draw(ai_b, board);
        }
        //  if (ret >= 0){
        //          small_reward(ai,score_board(board->board));            
        //  }
        free_board(board);
    }
}

int get_best_ai(AI_instance_t **ais, int n, int lim) {
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

void sighandler(int sig) {
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
    printf("answer was %s\n", buffer);
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

    exit(0);
}

void usage(char **argv, struct option *options) {
    int i;
    printf("USAGE: %s <options>\n", argv[0]);
    printf("Available options:\n");
    for (i = 0; options[i].name; i++)
        printf("    -%c, --%s %s\n", options[i].val, options[i].name,
            options[i].has_arg == required_argument ? "<argument>" : "");
}

void parse_arguments(int argc, char **argv) {
    int c;
    int option_index = 0;
    static struct option long_options[] = {
        {"threads", required_argument, NULL, 't'},
        {"jobs", required_argument, NULL, 'j'},
        {"file", required_argument, NULL, 'f'},
        {"games-to-play", required_argument, NULL, 'g'},
        {"iterations", required_argument, NULL, 'i'},
        {"games-pr-iteration", required_argument, NULL, 'n'},
        {"ports", required_argument, NULL, 'p'},
        {"moves", required_argument, NULL, 'm'},
        {"training-mode", required_argument, NULL, 'a'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    while ((c = getopt_long(argc, argv, "t:j:f:g:i:n:p:m:a:h", long_options,
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

void natural_selection(void) {
    /*
    int ai1 = random_int_r(0, nr_jobs - 1);
    int ai2 = random_int_r(0, nr_jobs - 1);
    float score_ai1 = get_score(games[ai1].ai);
    float score_ai2 = get_score(games[ai2].ai);
    if (score_ai1 > score_ai2) {
        printf("mutating ai%d (score %f, %d wins, %d losses)",
                ai2, get_score(games[ai2].ai),
                games[ai2].ai->nr_wins, games[ai2].ai->nr_losses);
        printf(" from ai%d (score %f, %d wins, %d losses)\n",
                ai1, get_score(games[ai1].ai),
                games[ai1].ai->nr_wins, games[ai1].ai->nr_losses);

        mutate(games[ai2].ai, games[ai1].ai, 0);
    } else if (score_ai2 > score_ai1) {
        printf("mutating ai%d (score %f, %d wins, %d losses)",
                ai1, get_score(games[ai1].ai),
                games[ai1].ai->nr_wins, games[ai1].ai->nr_losses);
        printf(" from ai%d (score %f, %d wins, %d losses)\n",
                ai2, get_score(games[ai2].ai),
                games[ai2].ai->nr_wins, games[ai2].ai->nr_losses);

        mutate(games[ai1].ai, games[ai2].ai, 0);
    }
     */

    //   best = get_best_ai(games, nr_jobs, -1);

    // printf("BEST: ai%d (score %f, %d wins, %d losses, wlr: %f\n)",
    //                best, get_score(games[best].ai),
    //               games[best].ai->nr_wins, games[best].ai->nr_losses, games[best].ai->nr_wins/(float)games[best].ai->nr_losses);

}

/*if (port_type[i] == 1) {
                if (nand256(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i], i > (nr_ports / 2), i < (nr_ports / 4)))
                    SetBit(V, i);
            } else if (port_type[i] == 3) {
                if (or256(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i], i > (nr_ports / 2), i < (nr_ports / 4)))
                    SetBit(V, i);
            } else if (port_type[i] == 2) {
                if (nor256(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i], i > (nr_ports / 2), i < (nr_ports / 4)))
                    SetBit(V, i);
            } else if (port_type[i] == 4) {
                if (and256(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i], i > (nr_ports / 2), i < (nr_ports / 4)))
                    SetBit(V, i);
 */
int print_brain(AI_instance_t *a1, int tmp) {
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

    //print_board(board->board);
    //i/8*16;
    //i%(8*16);
    //board[i/8*16][(i/8*16)/16]
    /* print integers and floats */
    for (i = 0; i < a1->nr_ports; i++) {

        for (j = 0; j < a1->nr_synapsis; j++) {
            if ((i > a1->nr_ports - 1 || (TestBit(a1->used_port, i) || a1->output_tag[0][i])) && TestBit(a1->brain[0][i], j)) {
                piece_t piece = 1 << ((j % (8 * 16)) % 16);

                fprintf(f, "p_%d -> p_%d\n", i, j);
                if (j >= a1->nr_ports + a1->board_size / 2)
                    fprintf(f, "p_%d[label=\"x:%d y:%d piece:%s\"]\n", j, j / (8 * 16), (j % (8 * 16)) / 16, piece_to_str(piece));
                else if (j >= a1->nr_ports)
                    fprintf(f, "p_%d[label=\"x:%d y:%d piece:%s(M)\"]\n", j, j / (8 * 16), (j % (8 * 16)) / 16, piece_to_str(piece));


                if (a1->port_type[0][i] == 1)
                    fprintf(f, "p_%d[color=red,style=filled,shape=octagon]\n", i);
                else if (a1->port_type[0][i] == 2)
                    fprintf(f, "p_%d[color=green,style=filled,shape=doublecircle]\n", i);
                else if (a1->port_type[0][i] == 3)
                    fprintf(f, "p_%d[color=chocolate2,style=filled,shape=star]\n", i);
                else if (a1->port_type[0][i] == 4)
                    fprintf(f, "p_%d[color=deeppink3,style=filled,shape=house]\n", i);
                if (a1->output_tag[0][i])
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

    printf("brain written to %s", filename);
    return 1;
}

int main(int argc, char *argv[]) {
    int j;

    parse_arguments(argc, argv);

    srand(100);

    init_threadpool(nr_threads);
    init_magicmoves();

    signal(SIGINT, sighandler);
    ais = malloc(nr_jobs * sizeof (struct AI_instance*));
    jobs = malloc(nr_jobs * sizeof (struct job));
    games = malloc(nr_jobs * sizeof (struct game_struct));

    for (i = 0; i < nr_jobs; i++) {
        if (ai_file) {
            ais[i] = load_ai(ai_file, mutation_rate);
            clear_score(ais[i]);
        } else
            ais[i] = ai_new(mutation_rate, brain_size, nr_ports); // mutation rate = 5000
        if (ais[i] == NULL) {
            perror("ai creation");
            exit(1);
        }

        games[i].ais = ais;
        //games[i].max_moves = max_moves;
        //        games[i].max_moves = 50;

        //games[i].do_a_move = do_nonrandom_move;
        //games[i].do_a_move = do_random_move;
        games[i].fen = DEFAULT_FEN;
        games[i].game_id = i;

        jobs[i].data = games + i;
        if (ai_vs_ai)
            jobs[i].task = play_chess_aivsai;
        else
            jobs[i].task = play_chess;

    }
    for (i = 0; i < nr_jobs; i++) {
        put_new_job(jobs + i);
    }
    iteration = 0;
    while (iteration < max_iterations) {
        printf("iteration %d\n", ++iteration);




        /* wait for jobs to finish */
        while (get_jobs_left() > 0 || get_jobs_in_progess() > 0)
            usleep(1000 * 1); // sleep 10 ms

        if (selection_function == 0) {
            for (j = 0; j < nr_selections; j++)
                natural_selection();
        }
        if (ai_vs_ai) {
            int nr_mutated = 0;

            printf("FIRST HALF: \n");
            best = get_best_ai(ais, nr_jobs / 2, -1);
            printf("best: %d\n", best);
            for (i = 0; i < nr_jobs / 2; put_new_job(jobs + i), i++) {
                if (games_to_play > ais[best]->nr_games_played) {
                    continue;
                }
                if (i == best) {
                    continue;
                }
                if (get_score(ais[i]) < get_score(ais[best])) {
                    mutate(ais[i], ais[best], 0, !nr_mutated);
                    nr_mutated++;

                }
            }
            printf("SECOND HALF: \n");
            best = get_best_ai(&ais[(nr_jobs / 2)], nr_jobs / 2, -1) + nr_jobs / 2;
            printf("best: %d\n", best);
            for (i = nr_jobs / 2; i < nr_jobs; put_new_job(jobs + i), i++) {
                if (games_to_play > ais[best]->nr_games_played)
                    break;
                if (i == best)
                    continue;
                if (get_score(ais[i]) < get_score(ais[best])) {
                    mutate(ais[i], ais[best], 0, !nr_mutated);
                    nr_mutated++;

                }
            }

        } else {
            int nr_mutated = 0;
            best = get_best_ai(ais, nr_jobs, -1);
            printf("best AI VS RANDOM: %d\n", best);
            if (!(iteration % 100)) {
                printf("print tmp brain\n");
                print_brain(ais[best], 1);
            }
            //printf("nr played %d\n", ais[best]->nr_games_played);
            for (i = 0; i < nr_jobs; put_new_job(jobs + i), i++) {
                if (games_to_play > ais[best]->nr_games_played) {

                    continue;
                }
                if (i == best) {
                    continue;
                }
                if (get_score(ais[i]) < get_score(ais[best])) {
                    mutate(ais[i], ais[best], 0, !nr_mutated);
                    nr_mutated++;
                }
            }
            printf("nr_muated: %f\n", ((float) nr_mutated) / nr_jobs);


        }
    }
    if (ai_vs_ai)
        best = get_best_ai(ais, nr_jobs / 2, -1);
    else
        best = get_best_ai(ais, nr_jobs, -1);

    print_brain(ais[best], 0);
    dump_ai("ai.aidump", ais[best]);
    clear_score(ais[best]);

    return 0;
}
