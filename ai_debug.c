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
#include "uci.h"

struct game_struct {
    AI_instance_t *ai;
    int games_to_play, max_moves;
    int (*do_a_move)(board_t *, struct uci *engine);
    char *fen;
    char *engine;

    int game_id;
};

int nr_jobs = 1, i, best, iteration;
struct game_struct *games = NULL;
struct job *jobs = NULL;

void play_chess(void *arg)
{
    struct game_struct *game = (struct game_struct *)arg;
    int games_to_play, nr_games, max_moves, moves, ret = -1;
    int (*do_a_move)(board_t *, struct uci *engine);
    struct uci *engine = NULL;
    AI_instance_t *ai;
    board_t *board;

    ai = game->ai;
    games_to_play = game->games_to_play;
    max_moves = game->max_moves;
    do_a_move = game->do_a_move;

    if (game->engine)
        engine = uci_init(game->engine, game->fen, BLACK);

    printf("starting game %d\n", game->game_id);

    for (nr_games = 0; nr_games < games_to_play; nr_games++) {
        board = new_board(game->fen);
        for (moves = 0; moves < max_moves; moves++) {
            print_board(board->board);
            getchar();

            ret = do_best_move(ai, board, NULL);
            if(ret == 0) {
                printf("stalemate\n");
                break;
            } else if (ret == -1) {
                printf("ai lost\n");
                punish(ai);
                break;
            }
            print_board(board->board);
            getchar();

            ret = do_a_move(board, engine);
            if(ret == 0) {
                printf("stalemate\n");
                break;
            } else if(ret == -1) {
                printf("ai won\n");
                reward(ai);
                break;
            }
        }

        if (ret >= 0)
            ++game->ai->nr_games_played;

        free_board(board);
    }
}

int main(int argc, char *argv[])
{
    char *ai_file = NULL;

    if (argc != 3) {
        printf("USAGE: %s --file <ai dump file>\n", argv[0]);
        exit(1);
    }
    if (!strcmp("--file", argv[1])) {
        ai_file = argv[2];
    }

    init_magicmoves();

    games = malloc(nr_jobs * sizeof(struct game_struct));

    i = 0;
    games[i].ai = load_ai(ai_file);
    clear_score(games[i].ai);
    if (games[i].ai == NULL) {
        perror("ai creation");
        exit(1);
    }

    games[i].engine = NULL;
    games[i].games_to_play = 1;
    games[i].max_moves = 20;
    games[i].do_a_move = do_random_move;
    //games[i].do_a_move = do_random_move;
    games[i].fen = DEFAULT_FEN;
    games[i].game_id = i + 1;

    play_chess(games);

    iteration = 0;

    return 0;
}
