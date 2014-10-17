#ifndef __AI_H
#define __AI_H

#include <stdint.h>
#include "common.h"
#include "board.h"

#define SetBit(A,k)     ((A)[((k)/32)] |= (1 << ((k)%32)))
#define ClearBit(A,k)   ((A)[((k)/32)] &= ~(1 << ((k)%32)))
#define TestBit(A,k)    ((A)[((k)/32)] & (1 << ((k)%32)))

typedef struct AI_instance {
    piece_t ***layers;
    int nr_layers, move_nr;
    int nr_wins, nr_losses;
    int generation;
    int feature_density;
    piece_t *nextboard;
    int **brain;
    int nr_ports;
    int nr_synapsis;
    int board_size;
} AI_instance_t;

extern AI_instance_t *ai_new(int nr_layers, int *nr_features, int feature_density);
extern void ai_free(AI_instance_t *ai);
extern int do_best_move(AI_instance_t *ai, board_t *board);
extern void punish(AI_instance_t *ai);
extern void reward(AI_instance_t *ai);
extern int get_score(AI_instance_t *ai);
extern void ai_mutate(AI_instance_t *a, AI_instance_t *b);
extern void clear_nr_wins(AI_instance_t *ai);

#endif
