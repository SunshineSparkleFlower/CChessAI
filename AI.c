#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "common.h"
#include "AI.h"
#include "map.h"
#include "rules.h"
#include "fastrules.h"

#include "emmintrin.h"
#include "smmintrin.h"

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

    /*
    ret->map = new_map(nr_features);
    if (ret->map == NULL) {
        perror("malloc");
        free(ret);
        return NULL;
    }

    ret->shortmemory = new_map(nr_features);
    if (ret->shortmemory == NULL) {
        perror("malloc");
        map_free(ret->map);
        free(ret);
        return NULL;
    }
    */

    ret->layers = malloc(nr_layers * sizeof(piece_t *));
    if (ret->layers == NULL) {
        perror("malloc");
        map_free(ret->shortmemory);
        free(ret->map);
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

    ret->nr_ports = 64;
    ret->board_size = 64*2*2*8;
    ret->nr_synapsis = ret->nr_ports + ret->board_size;

    //int *V = (int *)malloc((( nr_ports)/32)*sizeof(int)); 
    ret->brain = malloc_2d(ret->nr_synapsis/32, ret->nr_synapsis,  4);
    

    //int **board = (int *)malloc(64*2*2); 
    for (i = 0; i < ret->nr_synapsis; i++)
            random_fill(&ret->brain[i][0], ret->nr_synapsis/8);
/*
   ret->m = malloc(nr_features);
   if (ret->features == NULL) {
       perror("malloc");
       map_free(ret->shortmemory);
       free(ret->map);
       free(ret->m);
       free(ret);
       return NULL;
   }
*/

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
    map_free(ai->map);
    map_free(ai->shortmemory);

    free(ai->m);
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
    
   
        for (i = 0; likely(i < n); i++)
        { 

            if ((features[i] & board[i] & mask) == (board[i]) & mask){
                return 1;
            }    
        }
            return 0;
  
    
   
}
int multiply(piece_t *features, int *board, int n, int op, int mask)
{
    unsigned int i, s;
    
    int score = 0;
    if(op == 0){
        for (i = 0; likely(i < n); i++)
        { 
            if ((features[i] & board[i] & mask) == (board[i]) & mask){
                score++;
            }
            else{
                score--;
            }
        }
        return score+n;
    }
    else if(op == 1)
        for (i = 0; likely(i < n); i++){            
            if ((features[i] & board[i] & mask) != (features[i] & mask))
                return 0;
        }
    return 1;
}

int nand(int *a, int *b, int size, int *board, int board_size){

    int i;
    for(i = 0; i < size; i++){
        if(TestBit(a,i) == 1 && TestBit(b,i) == 1)
            return 0;
    }

    for(i = size; i < board_size+size; i++){
        if(TestBit(board,i) == 1 && TestBit(b,i) == 1)
            return 0;
    }
    return 1;
}





int eval_curcuit(int *V, int **M,  int nr_ports, int *board, int board_size){
    int i;
    for(i = 0; i < nr_ports; i++){
        if(nand(V, M[i], nr_ports, board, board_size))
            SetBit(V,i);
        else
            ClearBit(V,i);
    }
    return TestBit(V,(nr_ports-1))?1:0;
}


int score(AI_instance_t *ai, piece_t *board)
{
    int *V = (int *)malloc((( ai->nr_ports)/32)*sizeof(int)); 
    return eval_curcuit(V, ai->brain, ai->nr_ports, board, ai->board_size);
}







int _get_best_move(AI_instance_t *ai, board_t *board)
{
    int i, j, count;
    piece_t backup;
    float cumdist[board->moves_count], fcount, x;
    int scores[board->moves_count];

    for (i = count = 0; i < board->moves_count; i = count++) {
        memcpy(&board->board[64],&board->board[0], 64*sizeof(piece_t));

        do_move(board->board, board->moves[i].frm, board->moves[i].to, &backup);

        scores[i] = score(ai, board->board);
        reverse_move(board->board, board->moves[i].frm, board->moves[i].to, backup);
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

    get_all_legal_moves(board);
    if(board->moves_count == 0)
        return 0;
    if(board->moves_count == -1)
        return -1;
    
    best_move = _get_best_move(ai, board);
    do_move(board->board, board->moves[best_move].frm, board->moves[best_move].to, &backup);
 
     swapturn(board);
    return 1; 
}

//perform a random move return 0 if stalemate, -1 if check mate 1 of success
int do_random_move(board_t *board){
    piece_t backup;
    get_all_legal_moves(board);
     if(board->moves_count == 0)
        return 0;
    if(board->moves_count == -1)
        return -1;
     srand(time(NULL));
    int move = rand()%board->moves_count;
    do_move(board->board, board->moves[move].frm, board->moves[move].to, &backup);

   
     swapturn(board);

    return 1;

}

void punish(AI_instance_t *ai)
{
    map_weaken_axons(ai->map);
    ai->nr_losses++;
}

void reward(AI_instance_t *ai)
{
    map_strengthen_axons(ai->map);
    ai->nr_wins++;
}

/*
   void ai_mutate(AI_instance_t *a, AI_instance_t *b)
   {
   int feature, i;

   free(a->features);
   a->features = (piece_t ***)memdup_3d((void ***)b->features);
   if (a->features == NULL) {
   printf("failed to duplicate array!!\n");
   exit(1);
   }

   feature = 0;

   for (i = 0; i < a->feature_density; i++)
   feature |= piecesl[random_uint() % (sizeof(piecesl) / sizeof(piece_t))];

   a->features[random_uint() % b->nr_features][random_uint() % 8][random_uint() % 8]
   = feature;

   a->generation++;
   set_lr(a->map, b->map->lr_punish, b->map->lr_reward);
   mem_mutate(a->map);
   a->feature_density = b->feature_density;
   a->feature_density = MAX(a->feature_density + random_int_r(-1, 1), 0);
   }
   */

//layers in a a1 is replaced with layers from a2 pluss a mutation
int mutate(AI_instance_t *a1, AI_instance_t *a2){

      
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
