#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>
#include <ctype.h>
#include <string.h>

#include "common.h"
#include "board.h"
#include "AI.h"
#include "threadpool.h"

struct game_struct {
    AI_instance_t *ai;
    int games_to_play, max_moves;
    int (*do_a_move)(board_t *);
    char *fen;

    int game_id;
};

int nr_threads = 2, nr_jobs = 2, i, best, iteration;
struct game_struct *games = NULL;
struct job *jobs = NULL;

void print_ai_stats(int tid, AI_instance_t *ai, int ite, int rndwins)
{
    printf("thread %d: iteration %d\n", tid, ite);
    printf("thread %d: aiw nr wins: %d\n", tid, ai->nr_wins);
    printf("thread %d: random wins: %d\n", tid, rndwins);
    printf("thread %d: memory attributes:\n", tid);
    printf("    P and R\n");
    printf("thread %d: generation: %d\n", tid, ai->generation);
}

void play_chess(void *arg)
{
    struct game_struct *game = (struct game_struct *)arg;
    int games_to_play, nr_games, max_moves, moves, ret = -1;
    int (*do_a_move)(board_t *);
    AI_instance_t *ai;
    board_t *board;

    ai = game->ai;
    games_to_play = game->games_to_play;
    max_moves = game->max_moves;
    do_a_move = game->do_a_move;

    printf("starting game %d\n", game->game_id);

    for (nr_games = 0; nr_games < games_to_play; nr_games++) {
        board = new_board(game->fen);
        for (moves = 0; moves < max_moves; moves++) {
            ret = do_best_move(ai, board);
            if(ret == 0) {
                break;
            } else if (ret == -1) {
                punish(ai);
                break;
            }

            ret = do_a_move(board);
            if(ret == 0) {
                break;
            } else if(ret == -1) {
                reward(ai);
                break;
            }
        }

        if (ret >= 0)
            ++game->ai->nr_games_played;

        free_board(board);
    }
}

int get_best_ai(struct game_struct *g, int n)
{
    int i, best = 0;;

    for (i = 1; i < n; i++) {
        if (get_score(g[i].ai) > get_score(g[best].ai))
            best = i;
    }

    return best;
}

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
    fgets(buffer, sizeof(buffer), stdin);
    if (tolower(buffer[0]) != 'y')
        return;

    fprintf(stderr, "Save N best AIs to file? (Y/n): [default: y] ");
    fgets(buffer, sizeof(buffer), stdin);
    if (tolower(buffer[0]) == 'n')
        n = 0;

    if (n) {
        fprintf(stderr, "N (0-%d): [default: %d] ", nr_jobs, n);
        fgets(buffer, sizeof(buffer), stdin);
        buffer[127] = 0;
        if (buffer[0] != '\n')
            n = atoi(buffer);
        if (n > nr_jobs) {
            fprintf(stderr, "N > %d, setting N to %d\n", nr_jobs, nr_jobs);
            n = nr_jobs;
        }

        fprintf(stderr, "File prefix (<= 64 chars): [default: %s] ", file);
        fgets(buffer, 64, stdin);
        if (buffer[0] != '\n') {
            strncpy(file, buffer, 64);
            if (file[strlen(file) - 1] == '\n')
                file[strlen(file) - 1] = 0;
            file[64] = 0;
        }
    }

    for (i = 0; i < n; i++) {
        best = get_best_ai(games, nr_jobs);
        sprintf(buffer, "%s-%d.aidump", file, i + 1);
        fprintf(stderr, "saving ai to file '%s'\n", buffer);
        dump_ai(buffer, games[best].ai);
        clear_score(games[best].ai);
    }

    exit(0);
}

int main(int argc, char *argv[])
{
    char *ai_file = NULL;

    if (argc > 2 && !strcmp("--file", argv[1])) {
        ai_file = argv[2];
    } else {
        nr_threads = argc > 1 ? atoi(argv[1]) : 2;
        nr_jobs = argc > 2 ? atoi(argv[2]) : 2;
    }

    if (nr_threads == 0 || nr_jobs == 0) {
        printf("threads or jobs cannot be 0\n");
        printf("USAGE: %s <nr threads> <nr jobs>\n", argv[0]);
        return 1;
    }

    init_threadpool(nr_threads);
    init_magicmoves();

    signal(SIGINT, sighandler);

    jobs = malloc(nr_jobs * sizeof(struct job));
    games = malloc(nr_jobs * sizeof(struct game_struct));

    for (i = 0; i < nr_jobs; i++) {
        if (ai_file && i == 0) {
            games[i].ai = load_ai(ai_file);
            clear_score(games[i].ai);
        } else
            games[i].ai = ai_new();
        if (games[i].ai == NULL) {
            perror("ai creation");
            exit(1);
        }
        games[i].games_to_play = 1000;
        games[i].max_moves = 100;
        //games[i].do_a_move = do_nonrandom_move;
        games[i].do_a_move = do_random_move;
        games[i].fen = DEFAULT_FEN;
        games[i].game_id = i + 1;

        jobs[i].data = games + i;
        jobs[i].task = play_chess;
    }

    iteration = 0;
    while (1) {
        printf("iteration %d\n", ++iteration);
        for (i = 0; i < nr_jobs; i++) {
            put_new_job(jobs + i);
        }

        /* wait for jobs to finish */
        while (get_jobs_left() > 0 || get_jobs_in_progess() > 0)
            usleep(1000 * 10); // sleep 10 ms

        best = get_best_ai(games, nr_jobs);
        for (i = 0; i < nr_jobs; i++) {
            if (i == best)
                continue;

            printf("mutating ai%d (score %f, %d wins) from ai%d (score %f, %d wins)\n",
                    i, get_score(games[i].ai), games[i].ai->nr_wins, best, get_score(games[best].ai), games[best].ai->nr_wins);
            mutate(games[i].ai, games[best].ai);
        }
    }

    return 0;
}
