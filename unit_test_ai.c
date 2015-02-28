#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "AI.h"
#include "board.h"
#include "bitboard.h"

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

struct game_struct {
    int nr_games, thread_id;
    long checkmates, stalemates, timeouts;
};

void *moves_test(void *arg)
{
    int i, j, ret;
    board_t *board;
    long checkmate = 0, stalemate = 0, timeout = 0;
    struct game_struct *game = (struct game_struct *)arg;
    uint64_t num = 0, *num_moves;

    for (j = 0; j < game->nr_games; j++) {
        //board = new_board(DEFAULT_FEN);
        //board = new_board("rn2k2r/8/8/8/8/8/8/RN2K2R w KQkq - 0 1");
        board = new_board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        //board = new_board("Q6k/5K2/8/8/8/8/8/8 b - - 0 1");
        //board = new_board("Q6k/8/8/8/8/8/K7/6R1 b - - 0 1");

        for (i = 0; i < 100; i++) {
            printf("\n--------------- %s's turn --------------- \n", board->turn == WHITE ? "WHITE" : "BLACK");
            generate_all_moves(board);

            print_board(board->board);
            bb_print(board->white_pieces.apieces);
            printf("thread %d: --- %d %s ---\n", game->thread_id, i,
                    board->turn == WHITE ? "white" : "black");
            printf("legal_moves:\n");
            print_legal_moves(board);
            getchar();


            num += board->moves_count;
            ret = do_random_move(board);

            if (ret == 0) {
                debug_print("(%d) stalemate\n", game->thread_id);
                ++stalemate;
                break;
            } else if (ret == -1) {
                debug_print("(%d) checkmate\n", game->thread_id);
                ++checkmate;
                break;
            }
        }

        if (ret > 0)
            ++timeout;

        free(board);
    }

    game->checkmates = checkmate;
    game->stalemates = stalemate;
    game->timeouts = timeout;

    num_moves = malloc(sizeof(num));
    *num_moves = num;
    pthread_exit(num_moves);
}

uint64_t spawn_n_games(int n, int rounds)
{
    pthread_t threads[n - 1];
    int i;
    int checkmate, stalemate, timeout;
    struct game_struct games[n];
    uint64_t ret = 0, *tmp;

    for (i = 0; i < n; i++) {
        games[i].nr_games = rounds;
        games[i].thread_id = i + 1;

        pthread_create(&threads[i], NULL, moves_test, (void *)&games[i]);
    }

    checkmate = stalemate = timeout = 0;
    for (i = 0; i < n; i++) {
        pthread_join(threads[i], (void **)&tmp);
        ret += *tmp;
        free(tmp);

        checkmate += games[i].checkmates;
        stalemate += games[i].stalemates;
        timeout += games[i].timeouts;
    }

    printf("%d checkmates\n", checkmate);
    printf("%d stalemates\n", stalemate);
    printf("%d timeouts\n", timeout);

    return ret;
}

void random_test(void)
{
    int vals[11], ret;
    unsigned a = 9999999, i ;

    memset(vals, 0, sizeof vals);

    for (i = 0; i < a; i++) {
        ret = random_int_r(0, 10);
        if (ret < 0 || ret > 10) {
            printf("error: ret is = %d\n", ret);
            continue;
        }
        vals[ret]++;
    }

    for (i = 0; i < 11; i++)
        printf("%d: %d\n", i, vals[i]);
}

int main(int argc, char *argv[])
{
    //multiply_test();
    //score_test();
    //malloc_2d_test();
    //do_best_move_test();
    //mutate_test();

    init_magicmoves();

    //ai_dumptest();

    //ai_test();

    //random_test();

    unsigned long start, end;
    double diff;
    int rounds, threads, count;
    uint64_t num_trekk;

    start = now();
    rounds = argc > 1 ? atoi(argv[1]) : 1;
    threads = argc > 2 ? atoi(argv[2]) : 1;
    num_trekk = spawn_n_games(threads, rounds);
    end = now();

    diff = end - start;

    printf("%lu\n", num_trekk);
    count = rounds * threads;
    printf("%d games played in %.0f ms (%.1f games pr. second, w/ %d threads)",
            count, diff, (double)count / (diff / 1000), threads);
    printf(" %lu moves pr. second\n", (uint64_t)((double)num_trekk / (diff / 1000)));

    _shutdown();
    return 0;
}
