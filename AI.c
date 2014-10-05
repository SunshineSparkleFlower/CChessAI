#include <stdlib.h>
#include <stdint.h>
#include "common.h"
#include "rules.h"
#include "AI.h"
#include "map.h"

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

AI_instance_t *ai_new(int nr_features, int feature_density)
{
    int i;
    AI_instance_t *ret;
    uint16_t piecess[4096];
    uint16_t ***random_feature1;
    uint16_t ***random_feature2;
    uint16_t ***result_a, ***result_b;

    ret = malloc(sizeof(struct AI_instance));
    if (ret == NULL) {
        perror("malloc");
        return NULL;
    }

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

    ret->features = (piece_t ***)malloc_3d(8, 8, nr_features, sizeof(piece_t));
    if (ret->features == NULL) {
        perror("malloc");
        map_free(ret->shortmemory);
        free(ret->map);
        free(ret);
        return NULL;
    }

    ret->m = malloc(nr_features);
    if (ret->features == NULL) {
        perror("malloc");
        map_free(ret->shortmemory);
        free(ret->map);
        free(ret->m);
        free(ret);
        return NULL;
    }

    ret->move_nr = 0;
    ret->nr_wins = ret->nr_losses = 0;
    ret->generation = 0;
    ret->nr_features = nr_features;
    ret->feature_density = feature_density;

    for (i = 4096; i < 8192; i++)
        piecess[i - 4096] = i;

    random_feature1 = (uint16_t ***)malloc_3d(8, 8, nr_features, sizeof(uint16_t));
    random_feature2 = (uint16_t ***)malloc_3d(8, 8, nr_features, sizeof(uint16_t));

    result_a = (uint16_t ***)malloc_3d(8, 8, nr_features, sizeof(uint16_t));
    result_b = (uint16_t ***)malloc_3d(8, 8, nr_features, sizeof(uint16_t));

    choice_3d(piecess, sizeof(piecess) / sizeof(uint16_t), random_feature1);
    choice_3d(piecess, sizeof(piecess) / sizeof(uint16_t), random_feature2);

    int tmp;
    tmp = bitwise_or_3d((void ***)random_feature1, (void ***)random_feature2, (void ***)result_a);
    if (!tmp)
        debug_print("failed to bitwize or 1\n");

    choice_3d(piecess, sizeof(piecess) / sizeof(uint16_t), random_feature1);
    choice_3d(piecess, sizeof(piecess) / sizeof(uint16_t), random_feature2);

    tmp = bitwise_or_3d((void ***)random_feature1, (void ***)random_feature2, (void ***)result_b);
    if (!tmp)
        debug_print("failed to bitwize or 2\n");
    tmp = bitwise_or_3d((void ***)result_a, (void ***)result_b, (void ***)ret->features);
    if (!tmp)
        debug_print("failed to bitwize or 3\n");

    free(random_feature1);
    free(random_feature2);
    free(result_a);
    free(result_b);

    return ret;
}

void ai_free(AI_instance_t *ai)
{
    map_free(ai->map);
    map_free(ai->shortmemory);

    free(ai->m);
    free(ai->features);
    free(ai);
}

static int8_t multiply(piece_t *features, piece_t *board, int n)
{
    unsigned int i;
    __m128i ad, bd;

    for (i = 0; likely((i + ((128 / 8) / sizeof(piece_t))) < n);
            i += (128 / 8) / sizeof(piece_t)) {
        ad = _mm_loadu_si128((__m128i *)((piece_t *)features + i));
        bd = _mm_loadu_si128((__m128i *)((piece_t *)board + i));

        if (_mm_test_all_zeros(ad, bd))
            return 0;
    }

    for (; likely(i < n); ++i)
        if (features[i] != board[i])
            return 0;

    return 1;
}

static int _get_best_move(AI_instance_t *ai, board_t *board)
{
    int i, j, count;
    piece_t backup;
    float scores[board->moves_count];
    float cumdist[board->moves_count], fcount, x;

    for (i = count = 0; i < board->moves_count; i = count++) {
        do_move(board->board, board->moves[i].frm, board->moves[i].to, &backup);

        for (j = 0; j < ai->nr_features; j++)
            ai->m[j] = multiply(ai->features[j][0], board->_board, 64);

        scores[i] = map_lookup(ai->map, (char *)ai->m);
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

struct move *do_best_move(AI_instance_t *ai, board_t *board)
{
    int best_move, i;
    piece_t backup;

    get_all_legal_moves(board);
    best_move = _get_best_move(ai, board);

    do_move(board->board, board->moves[best_move].frm, board->moves[best_move].to, &backup);

    for (i = 0; i < ai->nr_features; i++)
        ai->m[i] = multiply(ai->features[i][0], board->board, 64);
    map_remember_action(ai->map, ai->m);

    return &board->moves[best_move];
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

int get_score(AI_instance_t *ai)
{
    return ai->nr_wins - ai->nr_losses;
}

void clear_nr_wins(AI_instance_t *ai)
{
    ai->nr_losses = ai->nr_wins = 0;
}
