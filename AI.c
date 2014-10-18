#include "smmintrin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "common.h"
#include "AI.h"


AI_instance_t *ai_new(void)
{
    int i;
    AI_instance_t *ret;

    ret = malloc(sizeof(struct AI_instance));
    if (ret == NULL) {
        perror("malloc");
        return NULL;
    }

    ret->nr_ports = 128;
    ret->board_size = 64*2*2*8;
    ret->nr_synapsis = ret->nr_ports + ret->board_size;

    ret->brain = (int **)malloc_2d(ret->nr_synapsis/32, ret->nr_synapsis,  4);


    random_fill(&ret->brain[0][0], (ret->nr_synapsis/32)*ret->nr_synapsis*4);
    for(i = 0; i < (ret->nr_synapsis/32)*ret->nr_synapsis*4*8; i++)
        if(!random_int_r(0, 100))
            SetBit(&ret->brain[0][0],i);
        else
            ClearBit(&ret->brain[0][0],i);

    ret->move_nr = 0;
    ret->nr_wins = ret->nr_losses = 0;
    ret->generation = 0;

    return ret;
}

void ai_free(AI_instance_t *ai)
{
    free(ai->brain);
    free(ai);
}

int nand(int *a, int *b, int size, piece_t *board, int board_size)
{
    __m128i ad, bd;

    ad = _mm_loadu_si128((__m128i *)a);
    bd = _mm_loadu_si128((__m128i *)b);
    if(!_mm_test_all_zeros(ad, bd))
        return 0;

    ad = _mm_loadu_si128((__m128i *)a);
    bd = _mm_loadu_si128((__m128i *)b);
    if(!_mm_test_all_zeros(ad, bd)){
        return 0;
    }

    int i;
    for (i = 0; i < (board_size)/32; i+=4) {
        ad = _mm_loadu_si128((__m128i *)(((int*)board)+i));
        bd = _mm_loadu_si128((__m128i *)((int*)b+(i+(size/32))));
        if(!_mm_test_all_zeros(ad, bd)){            
            return 0;
        }

    }
    return 1;

}

int nand_validation(int *a, int *b, int size, int *board, int board_size)
{

    int i;
    for (i = 0; i < size; i++) {
        if(TestBit(a,i)  && TestBit(b,i)){
            return 0;
        }
    }

    for (i = 0; i < board_size; i++) {
        if(TestBit(board,i)  && TestBit(b,(i+size))){
            return 0;
        }
    }
    return 1;
}

int eval_curcuit(int *V, int **M,  int nr_ports, piece_t *board, int board_size)
{   

    int i;
    for (i = 0; i < nr_ports; i++) {
        // if(nand(V, M[i], nr_ports, board, board_size) != nand_validation(V, M[i], nr_ports, board, board_size))
        //     printf("NAND and NAND validation did not return same value\n");


        if (nand(V, M[i], nr_ports, board, board_size))
            SetBit(V,i);
    }
    return !!TestBit(V,(nr_ports-1));
}

int score(AI_instance_t *ai, piece_t *board)
{
    int V[( ai->nr_ports)/32];

    bzero(V, sizeof(V));
    return eval_curcuit(V, ai->brain, ai->nr_ports, board, ai->board_size);
}

static int _get_best_move(AI_instance_t *ai, board_t *board)
{
    int i, count, moveret;
    float cumdist[board->moves_count], fcount, x;
    int scores[board->moves_count];

    printf("moves_count = %d\n", board->moves_count);

    memcpy(&board->board[64], &board->board[0], 64 * sizeof(piece_t));
    for (i = count = 0; i < board->moves_count; i = count++) {
        moveret = move(board, i);

        /* move returns 1 on success */
        if (moveret == 1) {
            scores[i] = score(ai, board->board);
            undo_move(board, i);
            continue;
        }

        /* move returns -1 if stalemate, 0 if i > board->moves_count */
        if (!moveret)
            break;
        else if (moveret == -1)
            return -1;
    }

    fcount = 0;
    for (i = 0; i < board->moves_count; i++) {
        fcount += scores[i];
        cumdist[i] = fcount;
    }
    printf("moves_count = %d\n", board->moves_count);
    x = random_float() * cumdist[board->moves_count - 1];
    printf("moves_count = %d\n", board->moves_count);

    printf("moves_count = %d, x = %f\n", board->moves_count, x);
    getchar();
    printf("x = %f, moves_count = %d\n", x, board->moves_count);
    getchar();

    return bisect(cumdist, x, board->moves_count);
}

//perform a best move return 0 if stalemate, -1 if check mate 1 of success
int do_best_move(AI_instance_t *ai, board_t *board)
{
    int best_move;

    generate_all_moves(board);
    if (is_checkmate(board))
        return -1;
    if (is_stalemate(board) || (best_move = _get_best_move(ai, board)) == -1)
        return 0;

    do_move(board, best_move);

    swapturn(board);
    return 1;
}

//perform a random move return 0 if stalemate, -1 if check mate 1 of success
int do_random_move(board_t *board)
{
    int rndmove;

    do {
        if (is_checkmate(board)) {
            debug_print("checkmate\n");
            return -1;
        }
        if (is_stalemate(board)) {
            debug_print("stalemate\n");
            return 0;
        }
        rndmove = random_int_r(0, board->moves_count - 1);

    } while (!do_move(board, rndmove));

    swapturn(board);

    return 1;
}

void punish(AI_instance_t *ai)
{
    ai->nr_losses++;
}

void reward(AI_instance_t *ai)
{
    ai->nr_wins++;
}

//layers in a a1 is replaced with layers from a2 pluss a mutation
int mutate(AI_instance_t *a1, AI_instance_t *a2)
{
    memcpy(&a1->brain[0][0], &a2->brain[0][0], a1->nr_synapsis* a1->nr_synapsis/32);

    unsigned r1 = random_uint()%a1->nr_synapsis;
    unsigned r2 = random_uint()%a1->nr_synapsis;
    SetBit(a1->brain[r1],r2);

    r1 = random_uint()%a1->nr_synapsis;
    r2 = random_uint()%a1->nr_synapsis;
    ClearBit(a1->brain[r1],r2);

    return 1;
}

int get_score(AI_instance_t *ai)
{
    return ai->nr_wins - ai->nr_losses;
}

void clear_nr_wins(AI_instance_t *ai)
{
    ai->nr_losses = ai->nr_wins = 0;
}
