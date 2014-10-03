#ifndef __MAP_H
#define __MAP_H

#include <cfuhash.h>

#include "list.h"
#include "common.h"

struct list_node {
    struct list_head list;
    void *data;
};

struct hmap {
    cfuhash_table_t *map;
    struct list_node shortmemory;
    float lr_punish, lr_reward;
    int nr_features;
};

extern struct hmap *new_map(int nr_features);
extern void map_free(struct hmap *map);
extern float map_lookup(struct hmap *map, char *key);
extern int map_remember_action(struct hmap *map, void *key);
extern float reward_func(struct hmap *map, float prew);
extern float punish_func(struct hmap *map, float prew);
extern void map_strengthen_axons(struct hmap *map);
extern void map_weaken_axons(struct hmap *map);
extern int get_len_memory(struct hmap *map);
extern void set_lr(struct hmap *map, float lr_p, float lr_r);
extern void mem_mutate(struct hmap *map);

#endif
