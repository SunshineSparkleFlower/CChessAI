#include "smmintrin.h"
#include "immintrin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "common.h"
#include "AI.h"

#define MAGIC_LENGTH 6
static unsigned char ai_mem_magic[] = "\x01\x02\x03\x04\x05\x06";

AI_instance_t *ai_new(int mutation_rate, int brain_size, int nr_ports) {
    int i, j, k;
    AI_instance_t *ret;

    ret = calloc(1, sizeof (struct AI_instance));
    if (ret == NULL) {
        perror("malloc");
        return NULL;
    }

    ret->nr_ports = nr_ports;
    ret->board_size = 64 * 2 * 2 * 8;
    ret->nr_synapsis = ret->nr_ports + ret->board_size;
    ret->nr_brain_parts = brain_size;
    ret->nr_outputs = 1;
    ret->brain = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->brain_a = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->brain_b = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    memset_3d((void ***) ret->brain_b, 0xff);

    //ret->correlations = (int ***) malloc_2d(2,ret->nr_ports, ret->nr_ports, sizeof (int));

    for (j = 0; j < ret->nr_brain_parts; j++) {
        for (i = 0; i < ret->nr_ports; i++) {
            for (k = 0; k < ret->nr_synapsis; k++) {

                //if (j == 0 && i == ret->nr_ports-1 && k == 0) {
                //     printf("happened\n");
                //      SetBit(ret->brain[j][i], k);
                // } else
                ClearBit(ret->brain[j][i], k);
            }
        }
    }

    ret->invert = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->output_tag = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->output_tag[0][ret->nr_ports - 1] = 1;
    ret->port_type = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    for (i = 0; i < ret->nr_brain_parts; i++) {
        for (j = 0; j < ret->nr_ports; j++) {
            ret->port_type[i][j] = random_int_r(1, 4);
            ret->invert[i][j] = random_int_r(0, 1);
        }
    }
    ret->nr_porttypes = 4;
    ret->mutationrate = (int**) malloc_2d(4, 2, sizeof (int));

    for (i = 0; i < 1; i++) {
        for (j = 0; j < ret->nr_porttypes; j++) {
            ret->mutationrate[i][j] = 1;
        }
    }

    for (i = 1; i < 2; i++) {
        for (j = 0; j < ret->nr_porttypes; j++) {
            ret->mutationrate[i][j] = 0;
        }
    }
    ret->activation_count = (int***) malloc_3d(2, ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->separation = (int***) malloc_3d(2, ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->separation_count = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    ret->state_separation = (int***) malloc_3d(2, ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->state_separation_count = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    ret->used_port = malloc(sizeof (int) * ret->nr_ports / 32);
    bzero(ret->used_port, sizeof (int) * ret->nr_ports / 32);

    ret->move_nr = 0;
    ret->nr_wins = ret->nr_losses = ret->nr_games_played = 0;
    ret->generation = 0;
    ret->mutation_rate = mutation_rate;

    /*
        for (i = 0; i < ret->nr_ports; i++) {
            int r_brain = random_int_r(0, ret->nr_brain_parts - 1);
            int r_port = i;
            ret->port_type[r_brain][r_port] = random_int_r(1, ret->nr_porttypes);
            ret->invert[r_brain][r_port] = random_int_r(0, 1);

            for (j = 0; j < ret->mutationrate[0][ret->port_type[r_brain][r_port]]; j++) {
                int r_synaps = random_int_r(ret->nr_ports, 256 + ret->nr_ports);

                SetBit(ret->brain[r_brain][r_port], r_synaps);
            }

        }
     */
    ret->zero_rate = 5;
    ret->one_rate = 5;
    ret->port_type_rate = 3;
    ret->unused_rate = 80;
    ret->output_rate = 30;
    ret->r_output_rate = 70;
    ret->separation_rate = 40;
    ret->state_separation_rate = 70;
    ret->separation_threshold = 35;
    ret->output_exponent = 5;
    ret->state_separation_threshold = 35;
    fprintf(stderr, "ret->brain %d\n", ret->nr_synapsis);
    dump(ret->brain[0][ret->nr_ports - 1], ret->nr_synapsis / 8);
    fprintf(stderr, "ret->brain_b: %p\n", ret->brain_b[0][0]);
    return ret;
}

/*AI_instance_t *copy_ai(AI_instance_t * ai) {
    AI_instance_t *ret = calloc(1, sizeof (AI_instance_t));

    memcpy(ret, ai, sizeof (AI_instance_t));
    ret->brain = (int ***) memdup_3d((void ***) ai->brain);

    clear_score(ret);
    return ret;
}
 */
int dump_ai(char *file, AI_instance_t * ai) {
    FILE *out;
    long brain_size = (ai->nr_synapsis / (sizeof (int) * 8)) *
            ai->nr_ports * sizeof (int)*ai->nr_brain_parts;

    out = fopen(file, "w");
    if (out == NULL)
        return 0;

    fwrite(ai_mem_magic, 1, MAGIC_LENGTH, out);
    fwrite(ai, 1, sizeof (AI_instance_t), out);

    fwrite(&ai->brain[0][0][0], 1, brain_size, out);
    fwrite(&ai->invert[0][0], 1, sizeof (int)* ai->nr_ports * ai->nr_brain_parts, out);

    fwrite(&ai->port_type[0][0], 1, ai->nr_ports * sizeof (int)*ai->nr_brain_parts, out);
    fwrite(&ai->mutationrate[0][0], 1, ai->nr_porttypes * 2 * sizeof (int), out);
    fwrite(&ai->output_tag[0][0], 1, sizeof (int)* ai->nr_ports * ai->nr_brain_parts, out);
    fwrite(&ai->used_port[0], 1, sizeof (int) * ai->nr_ports / 32, out);

    fclose(out);

    return 1;
}

AI_instance_t * load_ai(char *file, int mutation_rate) {
    int tmp;
    FILE *in;
    AI_instance_t *ret;
    unsigned char magic[MAGIC_LENGTH];
    long brain_size;
    ret = malloc(sizeof (AI_instance_t));

    in = fopen(file, "r");
    if (in == NULL) {
        perror("fopen");
        return NULL;
    }

    if (fread(magic, 1, MAGIC_LENGTH, in) != MAGIC_LENGTH) {
        fprintf(stderr, "Failed to read magic\n");
        perror("fread");
        return NULL;
    }
    if (memcmp(magic, ai_mem_magic, MAGIC_LENGTH)) {
        fprintf(stderr, "%s is not a valid AI dump\n", file);
        free(ret);
        fclose(in);
        return NULL;
    }

    if (fread(ret, 1, sizeof (AI_instance_t), in) != sizeof (AI_instance_t)) {
        fprintf(stderr, "Failed to read ai struct\n");
        perror("fread");
        return NULL;
    }
    ret->brain_a = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    ret->brain = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    ret->brain_b = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    memset_3d((void ***) ret->brain_b, 0xff);

    brain_size = (ret->nr_synapsis / (sizeof (int) * 8)) *
            ret->nr_ports * sizeof (int)*ret->nr_brain_parts;

    ret->invert = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->separation = (int***) malloc_3d(2, ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->separation_count = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->state_separation = (int***) malloc_3d(2, ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->state_separation_count = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    if (fread(&ret->brain[0][0][0], 1, brain_size, in) != brain_size) {
        fprintf(stderr, "Failed to read brain\n");
        perror("fread");
        return NULL;
    }
    if (fread(&ret->invert[0][0], 1, sizeof (int) * ret->nr_ports * ret->nr_brain_parts, in) !=
            sizeof (int) * ret->nr_ports * ret->nr_brain_parts) {
        fprintf(stderr, "Failed to read invert\n");
        perror("fread");
        return NULL;
    }

    ret->port_type = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    if (fread(&ret->port_type[0][0], 1, ret->nr_ports * sizeof (int)*ret->nr_brain_parts, in) !=
            ret->nr_ports * sizeof (int)*ret->nr_brain_parts) {
        fprintf(stderr, "Failed to read port type\n");
        perror("fread");
        return NULL;
    }

    ret->mutationrate = (int**) malloc_2d(ret->nr_porttypes, 2, sizeof (int));
    fprintf(stderr, "nr_porttypes: %d\n", ret->nr_porttypes);
    if (fread(&ret->mutationrate[0][0], 1, ret->nr_porttypes * 2 * sizeof (int), in) !=
            ret->nr_porttypes * 2 * sizeof (int)) {
        fprintf(stderr, "Failed to read mutationrate\n");
        perror("fread");
        return NULL;
    }
    ret->mutation_rate = mutation_rate;

    //ret->activation = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->activation_count = (int***) malloc_3d(2, ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    ret->output_tag = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    if ((tmp = fread(&ret->output_tag[0][0], 1, sizeof (int) * ret->nr_ports * ret->nr_brain_parts, in)) !=
            sizeof (int) * ret->nr_ports * ret->nr_brain_parts) {
        fprintf(stderr, "Failed to read output_tag. %d/%lu\n", tmp, sizeof (int) * ret->nr_ports * ret->nr_brain_parts);
        perror("fread");
        return NULL;
    }

    ret->used_port = malloc(sizeof (int) * ret->nr_ports / 32);
    if ((tmp = fread(&ret->used_port[0], 1, sizeof (int) * ret->nr_ports / 32, in)) !=
            sizeof (int) * ret->nr_ports / 32) {
        fprintf(stderr, "Failed to read used port %d/%lu\n", tmp, sizeof (int) * ret->nr_ports / 32);
        perror("fread");
        return NULL;
    }
    fclose(in);

    fprintf(stderr, "loading done \n");

    return ret;
}

void ai_free(AI_instance_t * ai) {
    free(ai->brain);
    free(ai);
}

int nor(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b) {
    __m128i ad, bd, cd;

    int i;
    for (i = 0; i < (nr_ports) / 32; i += 4) {


        ad = _mm_loadu_si128((__m128i *) (((int*) a) + i));
        bd = _mm_loadu_si128((__m128i *) (((int*) b) + i));
        cd = _mm_loadu_si128((__m128i *) (((int*) brain_a) + i));
        cd = _mm_or_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_a) + i), cd);


        cd = _mm_loadu_si128((__m128i *) (((int*) brain_b) + i));
        cd = _mm_and_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_b) + i), cd);

        if (!_mm_test_all_zeros(_mm_and_si128(ad, bd), bd)) {
            return 0;
        }
    }

    for (i = 0; i < (board_size) / 32; i += 4) {
        ad = _mm_loadu_si128((__m128i *) (((int*) board) + i));
        bd = _mm_loadu_si128((__m128i *) ((int*) b + (i + (nr_ports / 32))));
        cd = _mm_loadu_si128((__m128i *) (((int*) brain_a) + i + nr_ports / 32));
        cd = _mm_or_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_a) + i + nr_ports / 32), cd);


        cd = _mm_loadu_si128((__m128i *) (((int*) brain_b) + i + nr_ports / 32));
        cd = _mm_and_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_b) + i + nr_ports / 32), cd);


        if (!_mm_test_all_zeros(_mm_and_si128(ad, bd), bd)) {
            //  printf("here\n");
            return 0;
        }
    }
    return 1;
}

int nor256(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b, int ignore_board, int ignore_move) {
    __m256i ad, bd, and_res, board_d;


    int i;
    for (i = 0; i < (nr_ports) / 32; i += 8) {
        ad = _mm256_loadu_si256((__m256i *) (((int*) a) + i));
        bd = _mm256_loadu_si256((__m256i *) (((int*) b) + i));
        and_res = _mm256_and_si256(ad, bd);
        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i), cd);
         */

        if (!_mm256_testz_si256(and_res, bd)) {
            return 0;
        }
    }
    if (ignore_board)
        return 1;

    for (i = (board_size) / (32 * 2); i < (board_size) / 32; i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        and_res = _mm256_and_si256(ad, bd);
        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32), cd);
         */
        if (!_mm256_testz_si256(and_res, bd)) {
            return 0;
        }

    }
    if (ignore_move)
        return 1;
    for (i = 0; i < (board_size) / (32 * 2); i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        board_d = _mm256_loadu_si256((__m256i *) (((int*) board) + i + (board_size) / (32 * 2)));
        and_res = _mm256_and_si256(_mm256_xor_si256(ad, board_d), bd);
        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32), cd);
         */
        if (!_mm256_testz_si256(and_res, bd)) {
            return 0;
        }

    }
    return 1;

}

int nand(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b) {
    __m128i ad, bd, cd;

    int i;
    for (i = 0; i < (nr_ports) / 32; i += 4) {
        ad = _mm_loadu_si128((__m128i *) (((int*) a) + i));
        bd = _mm_loadu_si128((__m128i *) (((int*) b) + i));
        cd = _mm_loadu_si128((__m128i *) (((int*) brain_a) + i));
        cd = _mm_or_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_a) + i), cd);


        cd = _mm_loadu_si128((__m128i *) (((int*) brain_b) + i));
        cd = _mm_and_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_b) + i), cd);


        if (!_mm_test_all_zeros(_mm_xor_si128(_mm_and_si128(ad, bd), bd), bd)) {
            return 1;
        }
    }

    for (i = 0; i < (board_size) / 32; i += 4) {
        ad = _mm_loadu_si128((__m128i *) (((int*) board) + i));
        bd = _mm_loadu_si128((__m128i *) ((int*) b + (i + (nr_ports / 32))));
        cd = _mm_loadu_si128((__m128i *) (((int*) brain_a) + i + nr_ports / 32));
        cd = _mm_or_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_a) + i + nr_ports / 32), cd);


        cd = _mm_loadu_si128((__m128i *) (((int*) brain_b) + i + nr_ports / 32));
        cd = _mm_and_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_b) + i + nr_ports / 32), cd);

        if (!_mm_test_all_zeros(_mm_xor_si128(_mm_and_si128(ad, bd), bd), bd)) {
            //  printf("here\n");
            return 1;
        }

    }
    return 0;

}

int nand256(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b, int ignore_board, int ignore_move) {
    __m256i ad, bd, and_res, board_d;


    int i;
    for (i = 0; i < (nr_ports) / 32; i += 8) {
        ad = _mm256_loadu_si256((__m256i *) (((int*) a) + i));
        bd = _mm256_loadu_si256((__m256i *) (((int*) b) + i));
        and_res = _mm256_and_si256(ad, bd);
        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i), cd);
         */

        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 1;
        }
    }
    if (ignore_board)
        return 0;

    for (i = (board_size) / (32 * 2); i < (board_size) / 32; i += 8) {
        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        and_res = _mm256_and_si256(ad, bd);
        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32), cd);
         */
        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 1;
        }
    }

    if (ignore_move)
        return 0;

    for (i = 0; i < (board_size) / (32 * 2); i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        board_d = _mm256_loadu_si256((__m256i *) (((int*) board) + i + (board_size) / (32 * 2)));
        and_res = _mm256_and_si256(_mm256_xor_si256(ad, board_d), bd);

        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32), cd);
         */
        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 1;
        }

    }
    return 0;

}

int and(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b) {
    __m128i ad, bd, cd;
    //printf("in nand\n");
    //_dump(a,128/8);
    //_dump(b,128/8);

    //printf("\n");
    //   ad = _mm_loadu_si128((__m128i *)a);
    //   bd = _mm_loadu_si128((__m128i *)b);
    //   if(!_mm_test_all_zeros(ad, bd)){
    //    printf("here\n");
    //        return 0;
    //    }

    int i;
    for (i = 0; i < (nr_ports) / 32; i += 4) {
        ad = _mm_loadu_si128((__m128i *) (((int*) a) + i));
        bd = _mm_loadu_si128((__m128i *) (((int*) b) + i));
        cd = _mm_loadu_si128((__m128i *) (((int*) brain_a) + i));
        cd = _mm_or_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_a) + i), cd);


        cd = _mm_loadu_si128((__m128i *) (((int*) brain_b) + i));
        cd = _mm_and_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_b) + i), cd);

        if (!_mm_test_all_zeros(_mm_xor_si128(_mm_and_si128(ad, bd), bd), bd)) {
            return 0;
        }
    }

    for (i = 0; i < (board_size) / 32; i += 4) {
        ad = _mm_loadu_si128((__m128i *) (((int*) board) + i));
        bd = _mm_loadu_si128((__m128i *) ((int*) b + (i + (nr_ports / 32))));
        cd = _mm_loadu_si128((__m128i *) (((int*) brain_a) + i + nr_ports / 32));
        cd = _mm_or_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_a) + i + nr_ports / 32), cd);


        cd = _mm_loadu_si128((__m128i *) (((int*) brain_b) + i + nr_ports / 32));
        cd = _mm_and_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_b) + i + nr_ports / 32), cd);

        if (!_mm_test_all_zeros(_mm_xor_si128(_mm_and_si128(ad, bd), bd), bd)) {
            //  printf("here\n");
            return 0;
        }

    }
    return 1;

}

int and256(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b, int ignore_board, int ignore_move) {
    __m256i ad, bd, and_res, board_d;


    int i;
    for (i = 0; i < (nr_ports) / 32; i += 8) {
        ad = _mm256_loadu_si256((__m256i *) (((int*) a) + i));
        bd = _mm256_loadu_si256((__m256i *) (((int*) b) + i));
        and_res = _mm256_and_si256(ad, bd);
        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i), cd);
         */
        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 0;
        }
    }
    if (ignore_board)
        return 1;
    for (i = (board_size) / (32 * 2); i < (board_size) / 32; i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        and_res = _mm256_and_si256(ad, bd);
        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32), cd);
         */
        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            //  printf("here\n");
            return 0;
        }

    }
    if (ignore_move)
        return 1;
    for (i = 0; i < (board_size) / (32 * 2); i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        board_d = _mm256_loadu_si256((__m256i *) (((int*) board) + i + (board_size) / (32 * 2)));
        and_res = _mm256_and_si256(_mm256_xor_si256(ad, board_d), bd);

        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32), cd);
         */
        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            //  printf("here\n");
            return 0;
        }

    }
    return 1;

}

int or(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b) {
    __m128i ad, bd, cd;

    int i;
    for (i = 0; i < (nr_ports) / 32; i += 4) {
        ad = _mm_loadu_si128((__m128i *) (((int*) a) + i));
        bd = _mm_loadu_si128((__m128i *) (((int*) b) + i));
        cd = _mm_loadu_si128((__m128i *) (((int*) brain_a) + i));
        cd = _mm_or_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_a) + i), cd);


        cd = _mm_loadu_si128((__m128i *) (((int*) brain_b) + i));
        cd = _mm_and_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_b) + i), cd);

        if (!_mm_test_all_zeros(_mm_and_si128(ad, bd), bd)) {
            return 1;
        }
    }

    for (i = 0; i < (board_size) / 32; i += 4) {
        ad = _mm_loadu_si128((__m128i *) (((int*) board) + i));
        bd = _mm_loadu_si128((__m128i *) ((int*) b + (i + (nr_ports / 32))));
        cd = _mm_loadu_si128((__m128i *) (((int*) brain_a) + i + nr_ports / 32));
        cd = _mm_or_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_a) + i + nr_ports / 32), cd);


        cd = _mm_loadu_si128((__m128i *) (((int*) brain_b) + i + nr_ports / 32));
        cd = _mm_and_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_b) + i + nr_ports / 32), cd);

        if (!_mm_test_all_zeros(_mm_and_si128(ad, bd), bd)) {
            //  printf("here\n");
            return 1;
        }

    }
    return 0;
}

int or256(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b, int ignore_board, int ignore_move) {

    __m256i ad, bd, and_res, board_d;

    int i;
    for (i = 0; i < (nr_ports) / 32; i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) a) + i));
        bd = _mm256_loadu_si256((__m256i *) (((int*) b) + i));
        and_res = _mm256_and_si256(ad, bd);
        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i), cd);
         */
        if (!_mm256_testz_si256(and_res, bd)) {
            return 1;
        }
    }
    if (ignore_board) {
        return 0;
    }
    for (i = (board_size) / (32 * 2); i < (board_size) / 32; i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        and_res = _mm256_and_si256(ad, bd);
        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32), cd);
         */
        if (!_mm256_testz_si256(and_res, bd)) {
            return 1;
        }
    }
    if (ignore_move)
        return 0;

    for (i = 0; i < (board_size) / (32 * 2); i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        board_d = _mm256_loadu_si256((__m256i *) (((int*) board) + i + (board_size) / (32 * 2)));
        and_res = _mm256_and_si256(_mm256_xor_si256(ad, board_d), bd);

        /*
           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32));
           cd = _mm256_or_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_a) + i + nr_ports / 32), cd);

           cd = _mm256_loadu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32));
           cd = _mm256_and_si256(and_res, cd);
           _mm256_storeu_si256((__m256i *) (((int*) brain_b) + i + nr_ports / 32), cd);
         */
        if (!_mm256_testz_si256(and_res, bd)) {
            return 1;
        }
    }

    return 0;

}

int nand_validation(int *a, int *b, int size, int *board, int board_size) {

    int i;
    for (i = 0; i < size; i++) {
        if (TestBit(a, i) && TestBit(b, i)) {
            return 0;
        }
    }

    for (i = 0; i < board_size; i++) {
        if (TestBit(board, i) && TestBit(b, (i + size))) {
            return 0;
        }
    }
    return 1;
}

int eval_curcuit(int *V, int **M, int nr_ports, piece_t *board, int board_size,
        int* port_type, int **brain_a, int **brain_b, int **activation_count, int **state_separation, int *invert, int nr_outputs, int *output_tag, int *used_port) {

    int i;
    for (i = 0; i < nr_ports; i++) {
        if (TestBit(used_port, i) || output_tag[i]) {
            if (port_type[i] == 1) {
                if (nand256(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i], i > (nr_ports / 2), i < (nr_ports / 4)))
                    SetBit(V, i);
            } else if (port_type[i] == 3) {
                if (or256(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i], i > (nr_ports / 2), i < (nr_ports / 4)))
                    SetBit(V, i);
            } else if (port_type[i] == 2) {
                if (nor256(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i], i > (nr_ports / 2), i < (nr_ports / 4)))
                    SetBit(V, i);
            } else if (port_type[i] == 4) {
                if (and256(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i], i > (nr_ports / 2), i < (nr_ports / 4)))
                    SetBit(V, i);
            } else
                fprintf(stderr, "ERROR: port type error(%d)", port_type[i]);
        }
    }
    //log the output of each ports
    for (i = 0; i < nr_ports; i++) {
        activation_count[i][!!TestBit(V, i)] = 1;
        state_separation[i][!!TestBit(V, i)] = 1;

    }

    int sum = 0;
    //sum the values of ports tagged as output port
    for (i = 0; i < nr_ports; i++)
        sum += ((!!TestBit(V, (i))) && output_tag[i]);

    return sum;

}

//return score of board based on ai

int score(AI_instance_t *ai, piece_t * board) {
    int V[(ai->nr_ports) / 32];

    int score_sum = 0;
    int i;
    for (i = 0; i < ai->nr_brain_parts; i++) {
        bzero(V, sizeof (V));
        score_sum += eval_curcuit(
                V, ai->brain[i], ai->nr_ports, board, ai->board_size,
                ai->port_type[i], ai->brain_a[i], ai->brain_b[i],
                ai->separation[i], ai->state_separation[i], ai->invert[i], ai->nr_outputs, ai->output_tag[i], ai->used_port);
    }
    return score_sum;
}

int _get_best_move(AI_instance_t *ai, board_t * board) {
    int i, j, count, moveret;
    float cumdist[board->moves_count], fcount, x;
    int scores[board->moves_count];

    bzero(&ai->separation[0][0][0], 2 * ai->nr_brain_parts * ai->nr_ports * sizeof (int));

    //make a copy of the current board state
    memcpy(&board->board[64], &board->board[0], 64 * sizeof (piece_t));
    for (i = count = 0; i < board->moves_count; i = ++count) {


        moveret = move(board, i);
        if (i == board->moves_count) {
            //   print_board(&board->board[64]);
            //   print_board(&board->board[0]);
        }

        /* move returns 1 on success */
        if (moveret == 1) {
            //score the move
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

    //check if a port had constant output
    for (i = 0; i < ai->nr_brain_parts; i++) {
        for (j = 0; j < ai->nr_ports; j++) {
            ai->separation_count[i][j] += (ai->separation[i][j][0] && ai->separation[i][j][1]);
        }
    }
    //printf("separation_count: %d\n", ai->separation_count[0][ai->nr_ports-1]);
    fcount = 0;
    int best_i = 0;
    int best_val = 0;
    for (i = 0; i < board->moves_count; i++) {
        //                  printf("%d, ", scores[i]);

        //if (best_val == ai->nr_outputs)
        //    break;
        if (scores[i] > best_val) {
            best_val = scores[i];
            best_i = i;
        }
    }
    if (best_i == 0 && board->moves_count > 1 && best_val == scores[1])
        return random_int_r(0, board->moves_count - 1);
    //printf("best_i: %d", best_i);
    //    printf("\n");

    //return best_i;



    for (i = 0; i < board->moves_count; i++) {
        //      printf("%d, ", scores[i]);
        int sum = scores[i];
        for (j = 0; j < ai->output_exponent; j++)
            sum *= scores[i];
        fcount += sum;
        cumdist[i] = fcount;
    }
    //printf("\n");
    x = random_float() * cumdist[board->moves_count - 1];
    if (bisect(cumdist, x, board->moves_count) >= board->moves_count)
        fprintf(stderr, "INVALID MOVE RETURNED\n");
    //printf("ret: %d, moves_count: %d\n", bisect(cumdist, x, board->moves_count), board->moves_count);
    return bisect(cumdist, x, board->moves_count);
}

//perform a best move return 0 if stalemate, -1 if check mate 1 of success

int do_best_move(AI_instance_t *ai, board_t * board) {
    int best_move;

    generate_all_moves(board);
    if (is_checkmate(board))
        return -1;
    if (is_stalemate(board) || (best_move = _get_best_move(ai, board)) == -1) {

        if (is_checkmate(board))
            return -1;

        return 0;
    }
    if (best_move < 0)
        printf("best move: %d\n", best_move);
    int ret = do_move(board, best_move);
    if (!ret)
        fprintf(stderr, "ret %d\n", ret);
    swapturn(board);
    return 1;
}

//perform a random move return 0 if stalemate, -1 if check mate 1 of success

int do_random_move(board_t * board) {
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

//perform a nonrandom move return 0 if stalemate, -1 if check mate 1 of success

int do_nonrandom_move(board_t * board) {
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
        rndmove = 0;

    } while (!do_move(board, rndmove));

    swapturn(board);

    return 1;
}

void punish(AI_instance_t * ai) {
    ai->nr_losses++;
    ai->nr_games_played++;
    int i, j;

    for (i = 0; i < ai->nr_brain_parts; i++) {
        for (j = 0; j < ai->nr_ports; j++) {
            ai->state_separation_count[i][j] += (ai->state_separation[i][j][0] && ai->state_separation[i][j][1]);
        }
    }
    bzero(&ai->state_separation[0][0][0], 2 * ai->nr_brain_parts * ai->nr_ports * sizeof (int));

}

void small_punish(AI_instance_t * ai) {
    ai->nr_games_played++;
}

void small_reward(AI_instance_t *ai, int reward) {
    ai->positive_reward += reward;
}

void reward(AI_instance_t * ai) {
    ai->nr_wins += 1;
    ai->nr_games_played++;
    int i, j;

    for (i = 0; i < ai->nr_brain_parts; i++) {
        for (j = 0; j < ai->nr_ports; j++) {
            ai->state_separation_count[i][j] += (ai->state_separation[i][j][0] && ai->state_separation[i][j][1]);
        }
    }
    bzero(&ai->state_separation[0][0][0], 2 * ai->nr_brain_parts * ai->nr_ports * sizeof (int));

}

void draw(AI_instance_t *ai, board_t * board) {
    //small_reward(ai, score_board(board));
    int i, j;

    for (i = 0; i < ai->nr_brain_parts; i++) {
        for (j = 0; j < ai->nr_ports; j++) {
            ai->state_separation_count[i][j] += (ai->state_separation[i][j][0] && ai->state_separation[i][j][1]);
        }
    }
    bzero(&ai->state_separation[0][0][0], 2 * ai->nr_brain_parts * ai->nr_ports * sizeof (int));

    ai->nr_games_played++;

}

void copy_ai(AI_instance_t *from_ai, AI_instance_t * to_ai) {
    //copy a2 over to a1
    memcpy(&to_ai->brain[0][0][0], &from_ai->brain[0][0][0], to_ai->nr_brain_parts * to_ai->nr_ports * to_ai->nr_synapsis / 8);
    memcpy(&to_ai->port_type[0][0], &from_ai->port_type[0][0], to_ai->nr_ports * to_ai->nr_brain_parts * sizeof (int));
    memcpy(&to_ai->output_tag[0][0], &from_ai->output_tag[0][0], to_ai->nr_brain_parts * to_ai->nr_ports * sizeof (int));

    //copy mutation rates
    to_ai->output_rate = from_ai->output_rate;
    to_ai->r_output_rate = from_ai->r_output_rate;
    to_ai->unused_rate = from_ai->unused_rate;
    to_ai->zero_rate = from_ai->zero_rate;
    to_ai->one_rate = from_ai->one_rate;
    to_ai->port_type_rate = from_ai->port_type_rate;
    to_ai->separation_rate = from_ai->separation_rate;
    to_ai->state_separation_rate = from_ai->state_separation_rate;
    to_ai->separation_threshold = from_ai->separation_threshold;
    to_ai->state_separation_threshold = from_ai->state_separation_threshold;
    to_ai->output_exponent = from_ai->output_exponent;
}

void mutate_mutation_rates(AI_instance_t *ai) {
    int max_val = 100;
    int min_val = 0;

    ai->output_rate -= random_int_r(0, 1);
    ai->output_rate += random_int_r(0, 1);
    ai->output_rate = keep_in_range(ai->output_rate, min_val, max_val);


    ai->r_output_rate -= random_int_r(0, 1);
    ai->r_output_rate += random_int_r(0, 1);
    ai->r_output_rate = keep_in_range(ai->r_output_rate, min_val, max_val);

    ai->unused_rate -= random_int_r(0, 1);
    ai->unused_rate += random_int_r(0, 1);
    ai->unused_rate = keep_in_range(ai->unused_rate, min_val, max_val);

    ai->state_separation_rate -= random_int_r(0, 1);
    ai->state_separation_rate += random_int_r(0, 1);
    ai->state_separation_rate = keep_in_range(ai->state_separation_rate, min_val, max_val);

    ai->separation_rate -= random_int_r(0, 1);
    ai->separation_rate += random_int_r(0, 1);
    ai->separation_rate = keep_in_range(ai->separation_rate, min_val, max_val);

    ai->state_separation_threshold -= random_int_r(0, 1);
    ai->state_separation_threshold += random_int_r(0, 1);
    ai->state_separation_threshold = keep_in_range(ai->state_separation_threshold, min_val, max_val);

    ai->separation_threshold -= random_int_r(0, 1);
    ai->separation_threshold += random_int_r(0, 1);
    ai->separation_threshold = keep_in_range(ai->separation_threshold, min_val, max_val);

    ai->port_type_rate -= random_int_r(0, 1);
    ai->port_type_rate += random_int_r(0, 1);
    ai->port_type_rate = keep_in_range(ai->port_type_rate, min_val, max_val);

    ai->zero_rate -= random_int_r(0, 1);
    ai->zero_rate += random_int_r(0, 1);
    ai->zero_rate = keep_in_range(ai->zero_rate, min_val, max_val);

    ai->one_rate -= random_int_r(0, 1);
    ai->one_rate += random_int_r(0, 1);
    ai->one_rate = keep_in_range(ai->one_rate, min_val, max_val);

    ai->output_exponent -= random_int_r(0, 1);
    ai->output_exponent += random_int_r(0, 1);
    ai->output_exponent = keep_in_range(ai->output_exponent, min_val, max_val);

}

//brain in a a1 is replaced with brain from a2 pluss a mutation

int mutate(AI_instance_t *a1, AI_instance_t * a2, int print, int print_stats) {
    int i, j;
    int max_val = 100;
    int min_val = 0;
    copy_ai(a2, a1);
    mutate_mutation_rates(a1);

    //chance to set one of the n last ports as an output port
    if (random_int_r(min_val, max_val) > a1->output_rate) {
        a1->output_tag[0][random_int_r(a1->nr_ports - 50, a1->nr_ports - 1)] = 1;
    }


    //chance to remove the output tag
    if (random_int_r(min_val, max_val) > a1->r_output_rate) {
        a1->output_tag[0][random_int_r(a1->nr_ports - 50, a1->nr_ports - 1)] = 0;
    }

    a1->mutation_rate = a2->mutation_rate;
    if (print) {
        fprintf(stderr, "zero_rate: %d, one_rate: %d, port_type_rate: %d \n", a2->zero_rate, a2->one_rate, a2->port_type_rate);
        for (i = 0; i < a1->nr_brain_parts; i++) {
            for (j = 0; j < a1->nr_ports; j++) {
                fprintf(stderr, "%d: %d, %d o:%d type: %d, games: %d \n", j, a2->separation_count[i][j], a2->state_separation_count[i][j], a2->output_tag[i][j], a2->port_type[i][j], a2->nr_games_played);
            }
        }
    }
    if (print)
        fprintf(stderr, "\n");


    //find and mutate unused ports
    for (j = 0; j < a1->nr_ports; j++) {
        if ((!TestBit(a2->used_port, j)) && (!a2->output_tag[0][j])) {
            if (random_int_r(min_val, max_val) > a1->unused_rate) {

                //reset port
                bzero(a1->brain[0][j], a1->nr_synapsis / 8);
                int r_port = j;
                int r_synaps = random_int_r(0, a1->nr_synapsis - 1);
                int k;
                for (k = 0; k < 2; k++) {
                    if (r_port > a1->nr_ports / 2)
                        r_synaps = random_int_r(0, r_port - 1);
                    else
                        while (r_synaps >= j && r_synaps < a1->nr_ports)
                            r_synaps = random_int_r(0, a1->nr_synapsis - 1);

                    SetBit(a1->brain[0][r_port], r_synaps);
                }
                if (print)
                    fprintf(stderr, "%d(r), ", j);
            } else
                if (print)
                fprintf(stderr, "%d, ", j);
        } else
            if (print)
            fprintf(stderr, "%d(u), ", j);
    }

    //check for constant ports and reset them
    for (i = 0; i < a1->nr_brain_parts; i++) {
        for (j = a1->nr_ports / 4; j < a1->nr_ports; j++) {
            if (((a1->separation_threshold * a2->separation_count[i][j]) / (1 + a2->nr_games_played)) == 0) {
                if (random_int_r(min_val, max_val) > a1->separation_rate) {
                    if (print)
                        fprintf(stderr, "r_port: %d, ", j);
                    bzero(a1->brain[i][j], a1->nr_synapsis / 8);
                    a1->port_type[i][j] = random_int_r(1, a1->nr_porttypes);
                    a1->invert[i][j] = random_int_r(0, 1);
                    int r_brain = i;
                    int r_port = j;
                    int k;
                    int r_synaps = random_int_r(0, a1->nr_synapsis - 1);

                    for (k = 0; k < 2; k++) {
                        if (j > a1->nr_ports / 2)
                            r_synaps = random_int_r(0, j - 1);
                        else
                            while (r_synaps >= j && r_synaps < a1->nr_ports)
                                r_synaps = random_int_r(0, a1->nr_synapsis - 1);

                        SetBit(a1->brain[r_brain][r_port], r_synaps);
                    }
                }
            }
        }
    }
    if (print)
        fprintf(stderr, "\n");
    //check for low entropy in state separation ports
    for (i = 0; i < a1->nr_brain_parts; i++) {
        for (j = 0; j < a1->nr_ports / 4; j++) {
            if (((a1->state_separation_threshold * a2->state_separation_count[i][j]) / (1 + a2->nr_games_played)) == 0) {
                if (random_int_r(min_val, max_val) > a1->state_separation_rate) {
                    if (print)
                        printf("r_port: %d, ", j);
                    bzero(a1->brain[i][j], a1->nr_synapsis / 8);
                    a1->port_type[i][j] = random_int_r(1, a1->nr_porttypes);
                    a1->invert[i][j] = random_int_r(0, 1);
                    int r_brain = i;
                    int r_port = j;
                    int k;
                    int r_synaps = random_int_r(0, a1->nr_synapsis - 1);

                    for (k = 0; k < 2; k++) {
                        if (j > a1->nr_ports / 2)
                            r_synaps = random_int_r(0, j - 1);
                        else
                            while (r_synaps >= j && r_synaps < a1->nr_ports)
                                r_synaps = random_int_r(0, a1->nr_synapsis - 1);
                        SetBit(a1->brain[r_brain][r_port], r_synaps);
                    }
                }
            }
        }
    }
    if (print)
        fprintf(stderr, "\n");

    //completely randomize a port
    for (i = 0; i < a1->port_type_rate; i++) {
        int r_brain = random_int_r(0, a1->nr_brain_parts - 1);
        int r_port = random_int_r(0, a1->nr_ports - 1);
        a1->port_type[r_brain][r_port] = random_int_r(1, a1->nr_porttypes);
        a1->invert[r_brain][r_port] = random_int_r(0, 1);

        bzero(a1->brain[r_brain][r_port], a1->nr_synapsis / 8);
        int r_synaps = 0; //random_int_r(0, a1->nr_synapsis - 1);
        if (r_port > a1->nr_ports / 2)
            r_synaps = random_int_r(0, r_port - 1);
        else {
            r_synaps = random_int_r(0, a1->nr_synapsis - 1);
            while (r_synaps >= r_port && r_synaps < a1->nr_ports)
                r_synaps = random_int_r(0, a1->nr_synapsis - 1);
        }
        SetBit(a1->brain[r_brain][r_port], r_synaps);
        if (r_port > a1->nr_ports / 2)
            r_synaps = random_int_r(0, r_port - 1);
        else {
            r_synaps = random_int_r(0, a1->nr_synapsis - 1);
            while (r_synaps >= r_port && r_synaps < a1->nr_ports)
                r_synaps = random_int_r(0, a1->nr_synapsis - 1);
        }
        SetBit(a1->brain[r_brain][r_port], r_synaps);
    }

    //remove a random connection
    for (i = 0; i < a1->zero_rate; i++) {
        int r_brain = random_int_r(0, a1->nr_brain_parts - 1);
        int r_port = random_int_r(0, a1->nr_ports - 1);
        int r_synaps = 0;
        r_synaps = random_int_r(0, a1->nr_synapsis - 1); // if (r_port >= a1->nr_ports - 1 - 9 && r_synaps > a1->nr_ports - 1 - 9)

        ClearBit(a1->brain[r_brain][r_port], r_synaps);
    }

    //create a new random connections
    for (i = 0; i < a1->one_rate; i++) {
        int r_brain = random_int_r(0, a1->nr_brain_parts - 1);
        int r_port = random_int_r(0, a1->nr_ports - 1);
        int r_synaps = 0;

        if (r_port > a1->nr_ports / 2)
            r_synaps = random_int_r(0, r_port - 1);
        else {
            r_synaps = random_int_r(0, a1->nr_synapsis - 1);
            while (r_synaps >= r_port && r_synaps < a1->nr_ports)
                r_synaps = random_int_r(0, a1->nr_synapsis - 1);
        }
        SetBit(a1->brain[r_brain][r_port], r_synaps);
    }

    //find ports that are actually in use
    __m128i ad, bd;
    if (print)
        fprintf(stderr, "unused ports: \n");
    for (i = 0; i < (a1->nr_ports) / 32; i += 4) {

        bd = _mm_loadu_si128((__m128i *) (((int*) a1->brain[0][0]) + i));
        for (j = 0; j < a1->nr_ports; j++) {
            ad = _mm_loadu_si128((__m128i *) (((int*) a1->brain[0][j]) + i));
            bd = _mm_or_si128(ad, bd);
        }
        _mm_storeu_si128((__m128i *) (((int*) a1->used_port + i)), bd);
    }
    int k;
    for (k = 0; k < 7; k++) {
        for (i = 0; i < (a1->nr_ports) / 32; i += 4) {

            bd = _mm_loadu_si128((__m128i *) (((int*) a1->brain[0][0]) + i));
            for (j = 0; j < a1->nr_ports; j++) {
                if (TestBit(a1->used_port, j) || a1->output_tag[0][j]) {
                    ad = _mm_loadu_si128((__m128i *) (((int*) a1->brain[0][j]) + i));
                    bd = _mm_or_si128(ad, bd);
                }
            }
            _mm_storeu_si128((__m128i *) (((int*) a1->used_port + i)), bd);
        }
        //dump(a1->used_port, a1->nr_ports / 32);
    }


    //reset the new AI
    bzero(a1->separation_count[0], a1->nr_brain_parts * a1->nr_ports * sizeof (int));
    bzero(a1->state_separation_count[0], a1->nr_brain_parts * a1->nr_ports * sizeof (int));
    bzero(a1->activation_count[0][0], 2 * a1->nr_ports * a1->nr_brain_parts * sizeof (int));
    if (print_stats)
        fprintf(stderr, "score %f, %d w, %d games from score %f, %d w, %d games, one_rate: %d, zero_rate: %d, p_type_rate: %d, unused_rate: %d, o_rate: %d, r_o_rate: %d, separation_threshold: %d, state_separation_threshold: %d, separation_rate: %d, state_separation_rate: %d, exponent: %d\n",
            get_score(a1), a1->nr_wins, a1->nr_games_played, get_score(a2), a2->nr_wins, a2->nr_games_played, a2->one_rate, a2->zero_rate, a2->port_type_rate, a2->unused_rate, a2->output_rate, a2->r_output_rate, a2->separation_threshold, a2->state_separation_threshold, a2->separation_rate, a2->state_separation_rate, a2->output_exponent);
    clear_score(a1);


    return 1;
}

int mutate2(AI_instance_t *a1, AI_instance_t * a2) {
    int i, j;
    unsigned r1, r2;
    //memcpy(a1, a2, sizeof(AI_instance_t));


    memcpy(&a1->brain[0][0][0], &a2->brain[0][0][0], a1->nr_brain_parts * a1->nr_ports * a1->nr_synapsis / 8);
    memcpy(&a1->port_type[0][0], &a2->port_type[0][0], a1->nr_ports * a1->nr_brain_parts * sizeof (int));
    memcpy(&a1->mutationrate[0][0], &a2->mutationrate[0][0], 4 * 2 * sizeof (int));

    a1->port_type[random_int_r(0, a1->nr_brain_parts - 1)][random_int_r(0, a1->nr_ports - 1)] = random_int_r(1, 4);
    for (i = 0; i < /*a1->mutation_rate*/0; i++) {
        int r = random_int_r(0, a1->nr_brain_parts - 1);
        r1 = random_int_r(0, a1->nr_ports - 1);


        //bzero(a1->brain[r][r1], (a1->nr_synapsis/8));

        r2 = random_int_r(0, a1->nr_synapsis - 1);
        // printf("r: %d, r1: %d, r2: %d \n", r, r1,r2);
        //

        SetBit(a1->brain[r][r1], r2);
        //SetBit(a1->brain[r][r1], r2);
        for (j = 0; j < 100; j++) {
            int r = random_int_r(0, a1->nr_brain_parts - 1);
            r1 = random_int_r(0, a1->nr_ports - 1);
            r2 = random_int_r(0, a1->nr_synapsis - 1);
            ClearBit(a1->brain[r][r1], r2);
        }
        //       r = random_int_r(0,a1->nr_brain_parts-1);
        //       r1 = random_int_r(0,a1->nr_ports-1);


    }
    clear_score(a1);
    return 1;
}

int crossover(AI_instance_t *a1, AI_instance_t *a2, AI_instance_t * a3) {
    //int r = random_int_r(0,1);
    return -1;
}

int score_board(board_t * board) {
    int piece_score[13] = {1, 2, 2, 2, 3, 0,
        -1, -2, -2. - 2, -3, 0,
        0};
    int i;
    int score = 0;
    for (i = 0; i < 64; i++) {
        if (color(board->_board[i]) == WHITE)
            score += piece_score[get_moves_index(board->_board[i])];
        else if (color(board->_board[i]) == BLACK)
            score -= piece_score[get_moves_index(board->_board[i])];

        // printf("score: %d\n", score);
        // printf("move_index: %d\n", get_moves_index(board->_board[i]));

    }
    //print_board(board);
    // printf("score: %d\n", score);
    return score;

}

float get_score(AI_instance_t * ai) {
    int sum = 0;
    int i, j;
    for (i = 0; i < ai->nr_brain_parts; i++) {
        for (j = 0; j < ai->nr_ports; j++) {
            sum += ai->separation_count[i][j];
        }
    }
    //return sum;
    return ((float) (ai->nr_wins)) / ((float) ai->nr_games_played + 1); // + ai->positive_reward/((float)ai->nr_games_played+1);
    //return ((float)(ai->nr_wins))/((float)ai->nr_losses+1);
}

void clear_score(AI_instance_t * ai) {
    ai->nr_losses = ai->nr_wins = ai->nr_games_played = ai->positive_reward = 0;
}
