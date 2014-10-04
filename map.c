#include <stdlib.h>
#include <stdint.h>

#include "map.h"
#include "common.h"

static void map_free_function(void *data)
{
    free(data);
}

struct hmap *new_map(int nr_features)
{
    struct hmap *ret;

    ret = malloc(sizeof(struct hmap));
    if (ret == NULL)
        return NULL;

    ret->map = cfuhash_new_with_initial_size(64);
    cfuhash_set_free_function(ret->map, map_free_function);

    INIT_LIST_HEAD(&ret->shortmemory.list);

    ret->lr_punish = 0.1;
    ret->lr_reward = 1.0;
    ret->nr_features = nr_features;

    return ret;
}

void map_free(struct hmap *map)
{
    struct list_node *pos, *n;

    if (map == NULL)
        return;

    cfuhash_destroy(map->map);

    list_for_each_entry_safe(pos, n, &map->shortmemory.list, list) {
        list_del(&pos->list);
        free(pos);
    }

    free(map);
}

float map_lookup(struct hmap *map, char *key)
{
    float *ret = cfuhash_get_n(map->map, key, map->nr_features);
    if (ret == NULL) {
        //debug_print("FAILED TO LOOK UP VALUE IN MAP!\n");
        return 0.0;
    }

    return *ret;
}

int map_remember_action(struct hmap *map, void *key)
{
    struct list_node *node;

    node = malloc(sizeof(node));
    if (node == NULL)
        return 0;

    node->data = key;
    list_add_tail(&node->list, &map->shortmemory.list);

    return 1;
}

float reward_func(struct hmap *map, float prew)
{
    return MIN(prew * (1 - map->lr_reward) +  1 * (map->lr_reward), 1);
}

float punish_func(struct hmap *map, float prew)
{
    return MAX(prew * (1 - map->lr_punish) -  1 * (map->lr_punish), -1);
}

void map_strengthen_axons(struct hmap *map)
{
    float *prew, *tmp, deflt = 0.0;
    struct list_node *pos, *n;

    list_for_each_entry_safe_reverse(pos, n, &map->shortmemory.list, list) {
        tmp = cfuhash_get_n(map->map, pos->data, map->nr_features);
        if (tmp == NULL)
            tmp = &deflt;

        prew = malloc(sizeof(float));
        if (prew == NULL) {
            printf("%s failed to allocate memory\n", __FUNCTION__);
            return;
        }
        *prew = reward_func(map, *tmp);
        cfuhash_put_n(map->map, pos->data, map->nr_features, prew);
        list_del(&pos->list);
        free(pos);
    }
    debug_print("map size: %d\n", get_len_memory(map));
}

void map_weaken_axons(struct hmap *map)
{
    float *prew, *tmp, deflt = 0.0;
    struct list_node *pos, *n;

    list_for_each_entry_safe_reverse(pos, n, &map->shortmemory.list, list) {
        tmp = cfuhash_get_n(map->map, pos->data, map->nr_features);
        if (tmp == NULL)
            tmp = &deflt;

        prew = malloc(sizeof(float));
        if (prew == NULL) {
            printf("%s failed to allocate memory\n", __FUNCTION__);
            return;
        }
        *prew = punish_func(map, *tmp);
        cfuhash_put_n(map->map, pos->data, map->nr_features, prew);
        list_del(&pos->list);
        free(pos);
    }
}

int get_len_memory(struct hmap *map)
{
    return cfuhash_num_buckets_used(map->map);
}

void set_lr(struct hmap *map, float lr_p, float lr_r)
{
    map->lr_punish = lr_p;
    map->lr_reward = lr_r;
}

void mem_mutate(struct hmap *map)
{
    map->lr_punish *= 1 + random_int_r(-1, 1) / 100.0;
    map->lr_reward *= 1 + random_int_r(-1, 1) / 100.0;
}
