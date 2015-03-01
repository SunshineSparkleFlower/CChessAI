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
    struct game_struct *game = (struct game_struct *)arg;
    uint64_t num = 0, *num_moves;

    enum moves_index type_piece = EMPTY;

    for (j = 0; j < game->nr_games; j++) {
        board = new_board(DEFAULT_FEN);
        //board = new_board("rn2k2r/8/8/8/8/8/8/RN2K2R w KQkq - 0 1");
        //board = new_board("r3k2r/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/R3K2R w KQkq e6 0 1");
        //board = new_board("Q6k/5K2/8/8/8/8/8/8 b - - 0 1");
        //board = new_board("Q6k/8/8/8/8/8/K7/6R1 b - - 0 1");

        for (i = 0; i < 100; i++) {
            getchar();
            printf("\n--------------- %s's turn --------------- \n", board->turn == WHITE ? "WHITE" : "BLACK");
            print_board(board->board);
            bb_print(board->white_pieces.apieces);
            printf("thread %d: --- %d %s ---\n", game->thread_id, i,
                    board->turn == WHITE ? "white" : "black");

            ret = do_random_move_piece(board, type_piece);

            if (ret == 0) {
                printf("(%d) stalemate\n", game->thread_id);
                break;
            } else if (ret == -1) {
                printf("(%d) checkmate\n", game->thread_id);
                break;
            } else if (ret == -2) {
                printf("no more possible moves for piece type %d\n", type_piece);
                break;
            }
        }

        free_board(board);
    }

    num_moves = malloc(sizeof(num));
    *num_moves = num;
    pthread_exit(num_moves);
}

uint64_t spawn_n_games(int n, int rounds)
{
    pthread_t threads[n - 1];
    int i;
    struct game_struct games[n];
    uint64_t ret = 0, *tmp;

    for (i = 0; i < n; i++) {
        games[i].nr_games = rounds;
        games[i].thread_id = i + 1;

        pthread_create(&threads[i], NULL, moves_test, (void *)&games[i]);
    }

    for (i = 0; i < n; i++) {
        pthread_join(threads[i], (void **)&tmp);
        ret += *tmp;
        free(tmp);
    }

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


    int rounds, threads;

    rounds = argc > 1 ? atoi(argv[1]) : 1;
    threads = argc > 2 ? atoi(argv[2]) : 1;
    spawn_n_games(threads, rounds);

    _shutdown();
    return 0;
}
