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
    int rounds, max_moves;
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
    long rounds = game->rounds;
    int moves, max_moves = game->max_moves;
    int ai_wins = 0, ai_losses = 0;

    ai = ai_new();

    for (i = 0; i < rounds; i++) {
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

void spawn_n_games(int n, int rounds, int max_moves)
{
    pthread_t threads[n - 1];
    int i, ai_wins, ai_losses;
    int stalemate, timeout;
    struct game games[n];

    memset(games, 0, sizeof(games));

    for (i = 0; i < n; i++) {
        games[i].rounds = rounds;
        games[i].max_moves = max_moves;
        if (i == n - 1)
            break;
        pthread_create(&threads[i], NULL, ai_bench, (void *)&games[i]);
    }

    ai_bench((void *)&games[i]);
    ai_wins = games[i].ai_wins;
    ai_losses = games[i].ai_losses;
    stalemate = games[i].stalemates;
    timeout = games[i].timeouts;

    for (i = 0; i < n - 1; i++) {
        pthread_join(threads[i], NULL);

        ai_wins += games[i].ai_wins;
        ai_losses += games[i].ai_losses;
        stalemate += games[i].stalemates;
        timeout += games[i].timeouts;
    }

    printf("%d ai wins\n", ai_wins);
    printf("%d ai losses\n", ai_losses);
    printf("%d stalemates\n", stalemate);
    printf("%d timeouts\n", timeout);
}

int main(int argc, char *argv[])
{
    unsigned long start, end;
    double diff;
    int rounds, threads, count, max_moves;

    init_magicmoves();

    start = now();
    rounds = argc > 1 ? atoi(argv[1]) : 200;
    max_moves = argc > 2 ? atoi(argv[2]) : 100; 
    threads = argc > 3 ? atoi(argv[3]) : 1;

    spawn_n_games(threads, rounds, max_moves);
    end = now();

    diff = end - start;

    count = rounds * threads;
    printf("%d games played in %.0f ms (%.1f games pr. second, w/ %d threads, %d maximum moves)\n",
            count, diff, (double)count / (diff / 1000), threads, max_moves);
    return 0;
}
