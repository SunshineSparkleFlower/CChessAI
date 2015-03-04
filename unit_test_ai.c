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

static void print_stats(board_t *board)
{
    printf("------------- %s' turn -------------\n",
            board->turn == WHITE ? "white" : "black");
    print_board(board->board);
}

void uci_test(void)
{
    int i, j, ret;
    board_t *board;
    int num_games = 1;
    int max_moves = 50;
    struct uci *engine;

    engine = uci_init("/usr/games/stockfish", UCI_DEFAULT_FEN, BLACK);
    if (engine == NULL) {
        printf("Failed to initialize UCI engine!\n");
        return;
    }

    for (i = 0; i < num_games; i++) {
        board = new_board(DEFAULT_FEN);
        uci_new_game(engine, UCI_DEFAULT_FEN);
        for (j = 0; j < max_moves; j++) {
            ret = do_move_random_piece(board, engine);
#ifdef INSPECT_MOVES
            print_stats(board);
            getchar();
#endif

            if (ret == 0) {
                // stalemate
                break;
            } else if (ret == -1) {
                // checkmate
                printf("checkmate! uci won in %d moves\n", j);
                printf("fen: %s\n", get_fen(board));
                break;
            }

            ret = do_uci_move(board, engine);

#ifdef INSPECT_MOVES
            print_stats(board);
            getchar();
#endif
            if (ret == 0) {
                // stalemate
                break;
            } else if (ret == -1) {
                printf("checkmate! uci lost in %d moves\n", j);
                printf("fen: %s\n", get_fen(board));
                // checkmate
                break;
            }
        }
        free_board(board);
    }
    uci_close(engine);
}

void moves_test(char *fen, int num_games, int max_moves)
{
    int i, j, ret;
    board_t *board;

    for (i = 0; i < num_games; i++) {
        board = new_board(fen);
        for (j = 0; j < max_moves; j++) {
            ret = do_random_move(board, NULL);
#ifdef INSPECT_MOVES
            print_stats(board);
            getchar();
#endif
            if (ret == 0) {
                // stalemate
                break;
            } else if (ret == -1) {
                // checkmate
                break;
            }

            do_random_move(board, NULL);
#ifdef INSPECT_MOVES
            print_stats(board);
            getchar();
#endif
            if (ret == 0) {
                // stalemate
                break;
            } else if (ret == -1) {
                // checkmate
                break;
            }
        }
        free_board(board);
    }
}

void moves_consistency_test(void)
{
#ifndef BOARD_CONSISTENCY_TEST
    fprintf(stderr, "recompile with '-D BOARD_CONSISTENCY_TEST' to run this test!\n");
    return;
#endif
    moves_test(DEFAULT_FEN, 50000, 100);
}

void inspect_moves(void)
{
#ifndef INSPECT_MOVES
    fprintf(stderr, "recompile with '-D INSPECT_MOVES' to run this test!\n");
    return;
#endif
    moves_test("rnbqkbnr/pppp1ppp/8/8/8/8/PPPPpPPP/RNBQK2R w KQkq - 0 1", 1, 40);
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
    init_magicmoves();

    //random_test();
    //moves_consistency_test();
    inspect_moves();
    //uci_test();

    _shutdown();
    return 0;
}
