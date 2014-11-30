#ifndef __AI_H
#define __AI_H

#include <stdint.h>
#include "common.h"
#include "board.h"

#define SetBit(A,k)     ((A)[((k)/32)] |= (1 << ((k)%32)))
#define ClearBit(A,k)   ((A)[((k)/32)] &= ~(1 << ((k)%32)))
#define TestBit(A,k)    ((A)[((k)/32)] & (1 << ((k)%32)))

typedef struct AI_instance {
    int move_nr;
    int nr_wins, nr_losses, nr_games_played;
    float positive_reward;
    int generation;
    int ***brain;       
    int nr_ports;
    int nr_synapsis;
    int board_size;
    int mutation_rate;
    int nr_brain_parts;
    int *cu_brain;
    int iam_best;
    int *best_brain;
    piece_t *cu_board;
    struct cudaPitchedPtr *pitch;
    
    void *r_state;
} __attribute__((packed)) AI_instance_t;

#ifdef __cplusplus
extern "C" {
#endif

extern AI_instance_t *ai_new(int mutation_rate, int brain_size);
extern AI_instance_t *load_ai(char *file, int mutation_rate);
extern int dump_ai(char *file, AI_instance_t *ai);
extern void ai_free(AI_instance_t *ai);
extern AI_instance_t *copy_ai(AI_instance_t *ai);
extern int do_best_move(AI_instance_t *ai, board_t *board);
extern void punish(AI_instance_t *ai);
extern void reward(AI_instance_t *ai);
extern void draw(AI_instance_t *ai, board_t * board);
extern void cu_mutate(AI_instance_t *a1);
extern float get_score(AI_instance_t *ai);
extern int mutate(AI_instance_t *a1, AI_instance_t *a2);
extern void clear_score(AI_instance_t *ai);
extern int do_nonrandom_move(board_t *board);
extern int do_random_move(board_t *board);
extern int do_pseudo_random_move(board_t *board);


extern int score(AI_instance_t *ai, piece_t *board, int print);
extern int eval_curcuit(int *V, int **M,  int nr_ports, piece_t *board, int board_size);

#ifdef __cplusplus
}
#endif

#endif
