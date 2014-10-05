#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "rules.h"
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
    printf("    length %d\n", get_len_memory(ai->map));
    printf("    P and R\n");
    printf("    %f\n", ai->map->lr_punish);
    printf("    %f\n", ai->map->lr_reward);
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
            if (have_lost(board)) {
                print_board(board->board);
                printf("thread %d: ai lost\n", game->thread_id);
                punish(ai);
                ++rndwins;
                break;
            } else
                do_best_move(ai, board);

            swapturn(board);

            if (have_lost(board)) {
                print_board(board->board);
                printf("thread %d: ai won\n", game->thread_id);
                reward(ai);
                break;
            } else {
                get_all_legal_moves(board);
                rnd = random_int_r(0, board->moves_count - 1);
                move(board, rnd);
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
    int i;
    struct game_struct games[n];
    pthread_t threads[n - 1];

    for (i = 0; i < n; i++) {
        games[i].ai = ai_new(10, 4);
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


int oldmain(int argc, char *argv[])
{
    board_t *board = NULL;
    AI_instance_t *aiw[2];
    int i, tmp;
    int ite, turn, aiwon, randomwon;

    for (i = 0; i < 2; i++)
        aiw[i] = ai_new(10, 4); // 10 features, 4 feature density

    ite = 1;
    aiwon = 0;
    randomwon = 0;
    while (ite < 10000000) {
        turn = 0;
        board = new_board(NULL); // default board
        if (board == NULL) {
            printf("failed to allocate new board\n");
            continue;
        }

        if (ite % 100 == 0) {
            print_ai_stats(1, aiw[0], ite, randomwon);
            print_ai_stats(1, aiw[1], ite, randomwon);

            if (get_score(aiw[0]) > get_score(aiw[1])) {
                printf("0 won\n");
                ai_mutate(aiw[0], aiw[1]);
            } else {
                printf("1 won\n");
                ai_mutate(aiw[1], aiw[0]);
            }
            clear_nr_wins(aiw[0]);
            clear_nr_wins(aiw[1]);
            randomwon = 0;
        }

        swapturn(board);
        while (turn < 70) {
            swapturn(board);
            if (have_lost(board)) {
                randomwon++;
                print_board(board->board);
                printf("aiw lost\n");
                printf("%d\n", turn);
                punish(aiw[ite%2]);
                char *fen = get_fen(board);
                printf("fen: %s\n", fen);
                free(fen);
                break;
            }

            do_best_move(aiw[ite%2], board);
            swapturn(board);

            if (have_lost(board)) {
                aiwon++;
                print_legal_moves(board);
                print_board(board->board);
                reward(aiw[ite%2]);
                char *fen = get_fen(board);
                printf("fen: %s\n", fen);
                free(fen);
                printf("---------------------- AI WON ----------------------\n");
                break;
            }

            get_all_legal_moves(board);
            tmp = random_int_r(0, board->moves_count - 1);
            move(board, tmp);

            turn++;

        }
        if (turn == 70) {
            punish(aiw[ite % 2]);
            //debug_print("couldn't finish in < 70 turns\n");
            debug_print("giggles n shits (%d)\n", ite);
        }

        debug_print("more giggles n shits (%d)\n", ite);
        ite++;
        free_board(board);
    }

    for (i = 0; i < 2; i++)
        ai_free(aiw[i]);

    return 0;
}
