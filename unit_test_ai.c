#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>

#include "common.h"
#include "AI.h"
#include "board.h"
#include "bitboard.h"

#define MAX_WIDTH 70

unsigned long now(void)
{
    unsigned long ms;
    time_t s;
    struct timespec spec;

    clock_gettime(CLOCK_REALTIME, &spec);

    s = spec.tv_sec;
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
                char fen[1024];

                printf("checkmate! uci won in %d moves\n", j);
                get_fen(board, fen);
                printf("fen: %s\n", fen);
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
                char fen[1024];

                get_fen(board, fen);
                printf("fen: %s\n", fen);
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

int random_test(void)
{
    int success = 1;
    int vals[11], ret;
    unsigned a = 1000000, i;

    memset(vals, 0, sizeof vals);

    for (i = 0; i < a; i++) {
        ret = random_int_r(0, 10);
        if (ret < 0 || ret > 10) {
            printf("error: ret is = %d\n", ret);
            continue;
        }
        vals[ret]++;
    }
    int largest_val = vals[0];
    for (i = 0; i < 10; i++) {
        int val = abs(vals[i] - a / 10);
        if (val > largest_val) {
            largest_val = val;
        }
    }
    char buffer [250];
    sprintf(buffer, "Largest deviation in random distribution: %f", largest_val / (float) a);
    printf("%-*s  %s\n", MAX_WIDTH, buffer, largest_val / (float) a < 0.1 ? "OK" : "ERROR");
    success &= (largest_val / (float) a < 0.1);
    
    int error = 0;
    for (i = 0; i < 100000; i++) {
        if (random_int_r(-150, 150) > 150 || random_int_r(-150, 150)<-150)
            error = 1;
    }
    sprintf(buffer, "Random range test ");
    printf("%-*s  %s\n", MAX_WIDTH, buffer, !error ? "OK" : "ERROR");
    success &= !(error);

    return success;
}

int board_move_test()
{
    char buffer [250];

    board_t *board = new_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w aaaa - 0 1");
    generate_all_moves(board);
    sprintf(buffer, "Move count initial board test (count: %d)", board->moves_count);
    int error = board->moves_count != 20;
    printf("%-*s  %s\n", MAX_WIDTH, buffer, !error ? "OK" : "ERROR");
    return !error;
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
        {"random", no_argument, NULL, 'r'},
        {"board", no_argument, NULL, 'b'},
        {NULL, 0, NULL, 0},
    };
    int success = 1;
    while ((c = getopt_long(argc, argv, "brh", long_options,
            &option_index)) != -1)
        switch (c) {
            case 'r':
                printf("===============TESTING RANDOM FUNCTIONS===============\n\n");
                success &= random_test();
                break;
            case 'b':
                printf("================TESTING BOARD FUNCTIONS================\n\n");
                success &= board_move_test();
                break;
            case 'h':
            default:
                usage(argv, long_options);
                exit(0);
        }
    if (success) {
        printf("\nALL TESTS SUCCEDED\n");
    } else
        printf("\n ERROR: SOME TESTS FAILED\n");


}

int main(int argc, char *argv[])
{

    init_magicmoves();
    parse_arguments(argc, argv);

    //random_test();
    //moves_consistency_test();
    //inspect_moves();
    //uci_test();

    _shutdown();
    return 0;
}
