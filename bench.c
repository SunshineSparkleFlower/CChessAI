#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdio.h>
#include <pthread.h>
#include <strings.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "common.h"
#include "AI.h"
#include "board.h"

struct game {
    int thread_id;
    int max_moves, nr_games;
    int stalemates, timeouts, ai_wins, ai_losses;
};

unsigned long now(void)
{
    unsigned long ms;
    time_t s;
    struct timespec spec;

    clock_gettime(CLOCK_REALTIME, &spec);

    s  = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6); // convert nanoseconds to milliseconds

    return s * 1000 + ms;
}

void *ai_bench(void *arg)
{
    int i, ret = 1;
    board_t *board;
    AI_instance_t *ai;
    struct game *game = (struct game *)arg;
    long nr_games = game->nr_games;
    int moves, max_moves = game->max_moves;
    int ai_wins = 0, ai_losses = 0;
    int nr_ports = 512;
    ai = ai_new(nr_ports);

    for (i = 0; i < nr_games; i++) {
        board = new_board(NULL);
        for (moves = 0; moves < max_moves; moves++) {
            ret = do_best_move(ai, board);
            if (ret == -1) {
                ++ai_losses;
                break;
            } else if (ret == 0) {
                // break if stalemate or checkmate
                ++game->stalemates;
                break;
            }

            // break if stalemate or checkmate
            ret = do_random_move(board);
            if (ret == -1) {
                ++ai_wins;
                break;
            } else if (ret == 0) {
                // break if stalemate or checkmate
                ++game->stalemates;
                break;
            }
        }

        if (moves >= max_moves) {
            ++game->timeouts;
        }

        free_board(board);
    }
    ai_free(ai);

    game->ai_wins = ai_wins;
    game->ai_losses = ai_losses;

    return NULL;
}

void *move_gen(void *arg)
{
    int i, ret = 1;
    board_t *board;
    struct game *game = (struct game *)arg;
    long nr_games = game->nr_games;
    int moves, max_moves = game->max_moves;
    uint64_t num = 0, *num_moves;

    for (i = 0; i < nr_games; i++) {
        board = new_board(NULL);
        for (moves = 0; moves < max_moves; moves++) {
            generate_all_moves(board);
            num += board->moves_count;

            ret = do_random_move(board);
            if (ret <= 0)
                break;

            generate_all_moves(board);
            num += board->moves_count;

            ret = do_random_move(board);
            if (ret <= 0)
                break;
        }

        free_board(board);
    }

    num_moves = malloc(sizeof(num));
    *num_moves = num;
    pthread_exit(num_moves);
}

uint64_t spawn_n_games(int n, int nr_games, int max_moves, void *(*f)(void *))
{
    pthread_t threads[n - 1];
    int i;
    struct game games[n];
    uint64_t ret = 0, *tmp;

    for (i = 0; i < n; i++) {
        games[i].nr_games = nr_games;
        games[i].max_moves = max_moves;
        games[i].thread_id = i + 1;

        pthread_create(&threads[i], NULL, f, (void *)&games[i]);
    }

    for (i = 0; i < n; i++) {
        pthread_join(threads[i], (void **)&tmp);
        ret += *tmp;
        free(tmp);
    }

    return ret;
}

int move_gen_bench(int argc, char **argv)
{
    unsigned long start, end;
    double diff;
    uint64_t num_moves;
    int nr_games, threads, max_moves;

    init_magicmoves();

    start = now();
    nr_games = argc > 1 ? atoi(argv[1]) : 200;
    max_moves = argc > 2 ? atoi(argv[2]) : 100; 
    threads = argc > 3 ? atoi(argv[3]) : 1;
    num_moves = spawn_n_games(threads, nr_games, max_moves, move_gen);
    end = now();

    diff = end - start;

    printf("%lu moves in %.1f seconds = %lu moves pr. second. w/ %d threads\n",
            num_moves, diff / 1000, (uint64_t)((double)num_moves / (diff / 1000)), threads);

    return 0;
}

int main(int argc, char *argv[])
{
    init_magicmoves();

    move_gen_bench(argc, argv);

    return 0;
}
