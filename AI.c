#include "smmintrin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "common.h"
#include "AI.h"

//#include "smmintrin.h"

static piece_t piecesl[] = {
    WHITE_PAWN,
    WHITE_KNIGHT,
    WHITE_BISHOP,
    WHITE_ROOK,
    WHITE_QUEEN,
    WHITE_KING,
    BLACK_PAWN,
    BLACK_KNIGHT,
    BLACK_BISHOP,
    BLACK_ROOK,
    BLACK_QUEEN,
    BLACK_KING,
    P_EMPTY,
};

AI_instance_t *ai_new(int nr_layers, int *nr_features, int feature_density)
{
    int i,j;
    AI_instance_t *ret;
    uint16_t piecess[4096];
    uint16_t ***result_a, ***result_b;

    ret = malloc(sizeof(struct AI_instance));
    if (ret == NULL) {
        perror("malloc");
        return NULL;
    }

    ret->layers = malloc(nr_layers * sizeof(piece_t *));
    if (ret->layers == NULL) {
        perror("malloc");
        free(ret);
        return NULL;
    }

    for (i = 0; i < nr_layers; i++) {
        // every consecutive layer has twice the length of the previous one
        ret->layers[i] = (piece_t **)malloc_2d(i > 0 ? nr_features[i - 1] : 128,
                nr_features[i], sizeof(piece_t));

        ret->layers[i] = (piece_t **)malloc_2d(i > 0 ? nr_features[i - 1] : 128,
                nr_features[i], sizeof(piece_t));
        for (j = 0; j < nr_features[i]; j++)
            random_fill(&ret->layers[i][j][0],
                    i > 0 ? nr_features[i - 1] * sizeof(piece_t): 128 * sizeof(piece_t));
    }

    ret->nr_ports = 128;
    ret->board_size = 64*2*2*8;
    ret->nr_synapsis = ret->nr_ports + ret->board_size;

    //int *V = (int *)malloc((( nr_ports)/32)*sizeof(int));
    ret->brain = (int **)malloc_2d(ret->nr_synapsis/32, ret->nr_synapsis,  4);

    ret->move_nr = 0;
    ret->nr_wins = ret->nr_losses = 0;
    ret->generation = 0;
    ret->nr_layers = nr_layers;
    ret->feature_density = feature_density;

    return ret;
}

void ai_free(AI_instance_t *ai)
{
    int i;
    for (i = 0; i < ai->nr_layers; i++)
        free(ai->layers[i]);
    free(ai->layers);

    free(ai);
}

/*
int8_t multiply(piece_t *features, piece_t *board, int n)
{
    unsigned int i, s;
    __m128i ad, bd, tmp;

    for (i = 0; likely((i + ((128 / 8) / sizeof(piece_t))) < n);
            i += (128 / 8) / sizeof(piece_t)) {
        fprintf(stderr, "%s i = %d\n", __FUNCTION__, i);
        ad = _mm_loadu_si128((__m128i *)((piece_t *)features + i));
        bd = _mm_loadu_si128((__m128i *)((piece_t *)board + i));

        tmp = _mm_and_si128(ad, bd);
        tmp = _mm_xor_si128(ad, tmp);

        if (!_mm_test_all_zeros(tmp, bd))
            return 0;
    }

    if (i < n) {
        printf("%s shits wrong\n", __FUNCTION__);
        exit(1);
    }

    return 1;
}
*/

int multiply8(piece_t *features, piece_t *board, int n, int op, int mask)
{
    unsigned int i, s;

    for (i = 0; likely(i < n); ++i) {
        if ((features[i] & board[i] & mask) == (board[i]) & mask) {
            return 1;
        }
    }

    return 0;
}

int multiply(piece_t *features, int *board, int n, int op, int mask)
{
    unsigned int i, s;

    int score = 0;
    if (op == 0) {
        for (i = 0; likely(i < n); i++) {
            if ((features[i] & board[i] & mask) == (board[i]) & mask){
                score++;
            } else {
                score--;
            }
        }
        return score+n;
    } else if(op == 1)
        for (i = 0; likely(i < n); i++){
            if ((features[i] & board[i] & mask) != (features[i] & mask))
                return 0;
        }
    return 1;
}

int nand(int *a, int *b, int size, piece_t *board, int board_size)
{
     __m128i ad, bd, tmp;


        ad = _mm_loadu_si128((__m128i *)a);
        bd = _mm_loadu_si128((__m128i *)b);
        if(!_mm_test_all_zeros(ad, bd))
            return 0;
        
    int i;
    for (i = 0; i < (board_size)/32; i+=4) {
            ad = _mm_loadu_si128((__m128i *)(int*)board+i);
            bd = _mm_loadu_si128((__m128i *)(int*)b+i+(size/32));
            //printf("i: %d\n", i);
            
           if(!_mm_test_all_zeros(ad, bd))
                return 0;
    }
    return 1;

}

int eval_curcuit(int *V, int **M,  int nr_ports, piece_t *board, int board_size)
{
    int i;
    for (i = 0; i < nr_ports; i++) {
        if (nand(V, M[i], nr_ports, board, board_size))
            SetBit(V,i);
        else
            ClearBit(V,i);
    }
    return TestBit(V,(nr_ports-1)) ? 1 : 0;
}

int score(AI_instance_t *ai, piece_t *board)
{
    int ret, *V = (int *)malloc((( ai->nr_ports)/32)*sizeof(int));

    ret = eval_curcuit(V, ai->brain, ai->nr_ports, board, ai->board_size);

    free(V);
    return ret;
}

static int _get_best_move(AI_instance_t *ai, board_t *board)
{
    int i, j, count, moveret;
    piece_t backup;
    float cumdist[board->moves_count], fcount, x;
    int scores[board->moves_count];

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
    x = random_float() * cumdist[board->moves_count - 1];

    return bisect(cumdist, x, board->moves_count);
}

//perform a best move return 0 if stalemate, -1 if check mate 1 of success
int do_best_move(AI_instance_t *ai, board_t *board)
{
    int best_move, i;
    piece_t backup;

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
    piece_t backup;
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
}

int get_score(AI_instance_t *ai)
{
    return ai->nr_wins - ai->nr_losses;
}

void clear_nr_wins(AI_instance_t *ai)
{
    ai->nr_losses = ai->nr_wins = 0;
}
