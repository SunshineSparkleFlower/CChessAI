#ifndef __AI_H
#define __AI_H

#include <stdint.h>
#include "common.h"
#include "map.h"

typedef struct AI_instance {
    int8_t *m;
    piece_t ***features;
    int nr_features, move_nr;
    struct hmap *map, *shortmemory;
    int nr_wins, nr_losses;
    int generation;
    int feature_density;
} AI_instance_t;

extern AI_instance_t *ai_new(int nr_features, int feature_density);
extern void ai_free(AI_instance_t *ai);
extern struct move *do_best_move(AI_instance_t *ai, board_t *board);
extern void punish(AI_instance_t *ai);
extern void reward(AI_instance_t *ai);
extern int get_score(AI_instance_t *ai);
extern void ai_mutate(AI_instance_t *a, AI_instance_t *b);
extern void clear_nr_wins(AI_instance_t *ai);

#endif
