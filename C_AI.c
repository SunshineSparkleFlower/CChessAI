#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "common.h"
#include "board.h"
#include "AI.h"

struct game_struct {
    AI_instance_t *ai;
    int nr_games, thread_id;
};

void print_ai_stats(int tid, AI_instance_t *ai, int ite, int rndwins)
{
    printf("thread %d: iteration %d\n", tid, ite);
    printf("thread %d: aiw nr wins: %d\n", tid, ai->nr_wins);
    printf("thread %d: random wins: %d\n", tid, rndwins);
    printf("thread %d: memory attributes:\n", tid);
    printf("    P and R\n");
    printf("thread %d: generation: %d\n", tid, ai->generation);
    printf("density : %d\n", ai->feature_density);
}

void *play(void *arg)
{
    board_t *board;
    AI_instance_t *ai;
    int i, rounds, turn, rnd, rndwins;
    struct game_struct *game = (struct game_struct *)arg;

    rounds = game->nr_games;
    ai = game->ai;

    rndwins = 0;

    for (i = 1; i < rounds + 1; i++) {
        board = new_board(NULL);
        turn = 0;

        if (i % 100 == 0)
            print_ai_stats(game->thread_id, ai, i, rndwins);

        while (1) {
            if (is_checkmate(board)) {
                print_board(board->board);
                printf("thread %d: ai lost\n", game->thread_id);
                punish(ai);
                ++rndwins;
                break;
            } else
                do_best_move(ai, board);

            swapturn(board);

            if (is_checkmate(board)) {
                print_board(board->board);
                printf("thread %d: ai won\n", game->thread_id);
                reward(ai);
                break;
            } else {
                generate_all_moves(board);
                do {
                    rnd = random_int_r(0, board->moves_count - 1);
                } while (!move(board, rnd));
            }

            swapturn(board);
            ++turn;
            if (turn > 80)
                break;
        }

        free_board(board);
    }

    return NULL;
}

void spawn_n_games(int n, int rounds)
{
    pthread_t threads[n - 1];
    int i, num_layers = 3, ret;
    struct game_struct games[n];
    int features[] = {10, 100, 1}, feature_density = 10;

    for (i = 0; i < n; i++) {
        games[i].ai = ai_new(num_layers, features, feature_density);
        games[i].nr_games = rounds;
        games[i].thread_id = i + 1;

        if (i == n - 1)
            break;
        pthread_create(&threads[i], NULL, play, &games[i]);
    }

    play(&games[i]);
    ai_free(games[i].ai);

    for (i = 0; i < n - 1; i++) {
        pthread_join(threads[i], NULL);
        ai_free(games[i].ai);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("USAGE: %s <nr. concurrent games> <nr. rounds>\n", argv[0]);
        return 0;
    }

    spawn_n_games(atoi(argv[1]), atoi(argv[2]));
    return 0;
}
