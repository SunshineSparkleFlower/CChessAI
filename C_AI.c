#include <stdio.h>
#include <stdlib.h>
#include "rules.h"
#include "common.h"
#include "board.h"
#include "AI.h"

void print_mem_att(AI_instance_t *ai)
{
    printf("memory attributes:\n");
    printf("%d\n", get_len_memory(ai->map));
    printf("P and R\n");
    printf("%f\n", ai->map->lr_punish);
    printf("%f\n", ai->map->lr_reward);
}

void print_AI_att(AI_instance_t *ai)
{
    printf("mem att :\n");
    printf("density :\n");
    printf("%d\n", ai->feature_density);
}

int main(int argc, char *argv[])
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
        if (ite % 100 == 0) {
            printf("iteration %d\n", ite);
            printf("aiw[0] nr wins: %d\n", aiw[0]->nr_wins);
            printf("aiw[1] nr wins: %d\n", aiw[1]->nr_wins);
            printf("randomwon: %d\n", randomwon);
            print_mem_att(aiw[0]);
            print_mem_att(aiw[1]);
            printf("generation: %d\n", aiw[0]->generation);
            printf("generation: %d\n", aiw[1]->generation);
            print_AI_att(aiw[0]);
            print_AI_att(aiw[1]);

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
            //print_board(board->board);
            //printf("couldn't finish in < 50 turns\n");
        }

        ite++;
        free_board(board);
    }

    for (i = 0; i < 2; i++)
        ai_free(aiw[i]);

    return 0;
}
