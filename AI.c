#include "smmintrin.h"
#include "immintrin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "common.h"
#include "AI.h"
#include "bitboard.h"
#include "uci.h"

#define MAGIC_LENGTH 6
#define MIN_VAL 0
#define MAX_VAL 100
static unsigned char ai_mem_magic[] = "\x01\x02\x03\x04\x05\x06";

AI_instance_t *ai_new(int nr_ports)
{
    int j;
    AI_instance_t *ret;

    ret = calloc(1, sizeof (struct AI_instance));
    if (ret == NULL) {
        perror("malloc");
        return NULL;
    }

    ret->nr_ports = nr_ports;
    ret->low_port = nr_ports / 2;
    ret->high_port = nr_ports / 2;

    ret->board_size = 64 * 2 * 2 * 8;
    ret->nr_synapsis = ret->nr_ports + ret->board_size;
    ret->brain = (int **) malloc_2d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, sizeof (int));


    ret->invert = (int*) malloc(ret->nr_ports * sizeof (int));
    bzero(ret->invert, ret->nr_ports * sizeof (int));

    ret->output_tag = (int*) malloc(ret->nr_ports * sizeof (int));
    bzero(ret->output_tag, ret->nr_ports * sizeof (int));

    ret->nr_porttypes = 4;
    ret->port_type = (int*) malloc(ret->nr_ports * sizeof (int));
    for (j = 0; j < ret->nr_ports; j++) {
        ret->port_type[j] = random_int_r(1, ret->nr_porttypes);
        ret->invert[j] = random_int_r(0, 1);
    }



    ret->activation_count = (int**) malloc_2d(2, ret->nr_ports, sizeof (int));
    ret->separation = (int**) malloc_2d(2, ret->nr_ports, sizeof (int));
    ret->separation_count = (int*) malloc(ret->nr_ports * sizeof (int));
    ret->state_separation = (int**) malloc_2d(2, ret->nr_ports, sizeof (int));
    ret->state_separation_count = (int*) malloc(ret->nr_ports * sizeof (int));
    ret->used_port = malloc(sizeof (int) * ret->nr_ports / 32);
    bzero(ret->used_port, sizeof (int) * ret->nr_ports / 32);

    
    // initialize internal variables
    ret->move_nr = 0;
    ret->nr_wins = ret->nr_losses = ret->nr_games_played = 0;
    ret->generation = 0;
    ret->zero_rate = 10;
    ret->one_rate = 1;
    ret->port_type_rate = 2;
    ret->unused_rate = 90;
    ret->output_rate = 90;
    ret->r_output_rate = 90;
    ret->separation_rate = 50;
    ret->state_separation_rate = 95;
    ret->separation_threshold = 35;
    ret->output_exponent = 10;
    ret->state_separation_threshold = 35;
    //    create_random_connections(ret, 2,  a1->low_port, 0);

    return ret;
}

//write AI to file
int dump_ai(char *file, AI_instance_t * ai)
{
    FILE *out;
    long brain_size = (ai->nr_synapsis / (sizeof (int) * 8)) *
            ai->nr_ports * sizeof (int);

    out = fopen(file, "w");
    if (out == NULL)
        return 0;

    fwrite(ai_mem_magic, 1, MAGIC_LENGTH, out);
    fwrite(ai, 1, sizeof (AI_instance_t), out);

    fwrite(&ai->brain[0][0], 1, brain_size, out);
    fwrite(&ai->invert[0], 1, sizeof (int)* ai->nr_ports, out);

    fwrite(&ai->port_type[0], 1, ai->nr_ports * sizeof (int), out);
    fwrite(&ai->output_tag[0], 1, sizeof (int)* ai->nr_ports, out);
    fwrite(&ai->used_port[0], 1, sizeof (int) * ai->nr_ports / 32, out);

    fclose(out);

    return 1;
}

//load AI from file
AI_instance_t * load_ai(char *file)
{
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


    brain_size = (ret->nr_synapsis / (sizeof (int) * 8)) *
            ret->nr_ports * sizeof (int);


    ret->brain = (int **) malloc_2d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, sizeof (int));
    ret->invert = (int*) malloc(ret->nr_ports * sizeof (int));
    ret->output_tag = (int*) malloc(ret->nr_ports * sizeof (int));
    ret->port_type = (int*) malloc(ret->nr_ports * sizeof (int));
    ret->activation_count = (int**) malloc_2d(2, ret->nr_ports, sizeof (int));
    ret->separation = (int**) malloc_2d(2, ret->nr_ports, sizeof (int));
    ret->separation_count = (int*) malloc(ret->nr_ports * sizeof (int));
    ret->state_separation = (int**) malloc_2d(2, ret->nr_ports, sizeof (int));
    ret->state_separation_count = (int*) malloc(ret->nr_ports * sizeof (int));
    ret->used_port = malloc(sizeof (int) * ret->nr_ports / 32);





    if (fread(&ret->brain[0][0], 1, brain_size, in) != brain_size) {
        fprintf(stderr, "Failed to read brain\n");
        perror("fread");
        return NULL;
    }
    if (fread(&ret->invert[0], 1, sizeof (int) * ret->nr_ports, in) !=
            sizeof (int) * ret->nr_ports) {
        fprintf(stderr, "Failed to read invert\n");
        perror("fread");
        return NULL;
    }

    if (fread(&ret->port_type[0], 1, ret->nr_ports * sizeof (int), in) !=
            ret->nr_ports * sizeof (int)) {
        fprintf(stderr, "Failed to read port type\n");
        perror("fread");
        return NULL;
    }


    if ((tmp = fread(&ret->output_tag[0], 1, sizeof (int) * ret->nr_ports, in)) !=
            sizeof (int) * ret->nr_ports) {
        fprintf(stderr, "Failed to read output_tag. %d/%lu\n", tmp, sizeof (int) * ret->nr_ports);
        perror("fread");
        return NULL;
    }

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

void ai_free(AI_instance_t * ai)
{
    free(ai->brain);
    free(ai);
}

int nor(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b)
{
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

int nor256(int *a, int *b, int nr_ports, piece_t *board, int board_size)
{
    __m256i ad, bd, and_res, board_d;


    int i;
    for (i = 0; i < (nr_ports) / 32; i += 8) {
        ad = _mm256_loadu_si256((__m256i *) (((int*) a) + i));
        bd = _mm256_loadu_si256((__m256i *) (((int*) b) + i));
        and_res = _mm256_and_si256(ad, bd);


        if (!_mm256_testz_si256(and_res, bd)) {
            return 0;
        }
    }


    for (i = (board_size) / (32 * 2); i < (board_size) / 32; i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        and_res = _mm256_and_si256(ad, bd);

        if (!_mm256_testz_si256(and_res, bd)) {
            return 0;
        }

    }

    for (i = 0; i < (board_size) / (32 * 2); i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        board_d = _mm256_loadu_si256((__m256i *) (((int*) board) + i + (board_size) / (32 * 2)));
        and_res = _mm256_and_si256(_mm256_xor_si256(ad, board_d), bd);

        if (!_mm256_testz_si256(and_res, bd)) {
            return 0;
        }

    }
    return 1;

}

int nand(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b)
{
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

int nand256(int *a, int *b, int nr_ports, piece_t *board, int board_size)
{
    __m256i ad, bd, and_res, board_d;


    int i;
    for (i = 0; i < (nr_ports) / 32; i += 8) {
        ad = _mm256_loadu_si256((__m256i *) (((int*) a) + i));
        bd = _mm256_loadu_si256((__m256i *) (((int*) b) + i));
        and_res = _mm256_and_si256(ad, bd);

        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 1;
        }
    }


    for (i = (board_size) / (32 * 2); i < (board_size) / 32; i += 8) {
        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        and_res = _mm256_and_si256(ad, bd);

        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 1;
        }
    }



    for (i = 0; i < (board_size) / (32 * 2); i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        board_d = _mm256_loadu_si256((__m256i *) (((int*) board) + i + (board_size) / (32 * 2)));
        and_res = _mm256_and_si256(_mm256_xor_si256(ad, board_d), bd);

        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 1;
        }

    }
    return 0;

}

int and(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b)
{
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

int and256(int *a, int *b, int nr_ports, piece_t *board, int board_size)
{
    __m256i ad, bd, and_res, board_d;


    int i;
    for (i = 0; i < (nr_ports) / 32; i += 8) {
        ad = _mm256_loadu_si256((__m256i *) (((int*) a) + i));
        bd = _mm256_loadu_si256((__m256i *) (((int*) b) + i));
        and_res = _mm256_and_si256(ad, bd);

        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 0;
        }
    }

    for (i = (board_size) / (32 * 2); i < (board_size) / 32; i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        and_res = _mm256_and_si256(ad, bd);

        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 0;
        }

    }

    for (i = 0; i < (board_size) / (32 * 2); i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        board_d = _mm256_loadu_si256((__m256i *) (((int*) board) + i + (board_size) / (32 * 2)));
        and_res = _mm256_and_si256(_mm256_xor_si256(ad, board_d), bd);


        if (!_mm256_testz_si256(_mm256_xor_si256(and_res, bd), bd)) {
            return 0;
        }

    }
    return 1;

}

int or(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b)
{
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

int or256(int *a, int *b, int nr_ports, piece_t *board, int board_size)
{

    __m256i ad, bd, and_res, board_d;

    int i;
    for (i = 0; i < (nr_ports) / 32; i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) a) + i));
        bd = _mm256_loadu_si256((__m256i *) (((int*) b) + i));
        and_res = _mm256_and_si256(ad, bd);

        if (!_mm256_testz_si256(and_res, bd)) {
            return 1;
        }
    }

    for (i = (board_size) / (32 * 2); i < (board_size) / 32; i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        and_res = _mm256_and_si256(ad, bd);

        if (!_mm256_testz_si256(and_res, bd)) {
            return 1;
        }
    }


    for (i = 0; i < (board_size) / (32 * 2); i += 8) {

        ad = _mm256_loadu_si256((__m256i *) (((int*) board) + i));
        bd = _mm256_loadu_si256((__m256i *) ((int*) b + (i + (nr_ports / 32))));
        board_d = _mm256_loadu_si256((__m256i *) (((int*) board) + i + (board_size) / (32 * 2)));
        and_res = _mm256_and_si256(_mm256_xor_si256(ad, board_d), bd);

        if (!_mm256_testz_si256(and_res, bd)) {
            return 1;
        }
    }

    return 0;

}

int nand_validation(int *a, int *b, int size, int *board, int board_size)
{

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

int eval_curcuit(piece_t *board, AI_instance_t *ai)
{
    int i;
    int V[(ai->nr_ports) / 32];
    bzero(V, sizeof (V));

    for (i = ai->low_port; i <= ai->high_port; i++) {
        // if (ai->output_tag[i] || TestBit(ai->used_port, i)) {

        //printf("port: %d", i);
        if (ai->port_type[i] == 1) {
            if (nand256(V, ai->brain[i], ai->nr_ports, board, ai->board_size))
                SetBit(V, i);
        } else if (ai->port_type[i] == 3) {
            if (or256(V, ai->brain[i], ai->nr_ports, board, ai->board_size))
                SetBit(V, i);
        } else if (ai->port_type[i] == 2) {
            if (nor256(V, ai->brain[i], ai->nr_ports, board, ai->board_size))
                SetBit(V, i);
        } else if (ai->port_type[i] == 4) {
            if (and256(V, ai->brain[i], ai->nr_ports, board, ai->board_size))
                SetBit(V, i);
        } else
            fprintf(stderr, "ERROR: port type error(%d)", ai->port_type[i]);
        //}
    }

    //log the output of each ports
    for (i = ai->low_port; i <= ai->high_port; i++) {
        ai->activation_count[i][!!TestBit(V, i)] = 1;
        ai->state_separation[i][!!TestBit(V, i)] = 1;
    }

    int sum = 0;
    //sum the values of ports tagged as output port
    for (i = ai->low_port; i <= ai->high_port; i++)
        sum += ((!!TestBit(V, (i))) && ai->output_tag[i]);

    return sum;

}

int _get_best_move(AI_instance_t *ai, board_t * board)
{
    int i, j, count, moveret;
    float cumdist[board->moves_count], fcount, x;
    int scores[board->moves_count];

    bzero(&ai->separation[0][0], 2 * ai->nr_ports * sizeof (int));

    //make a copy of the current board state
    memcpy(&board->board[64], &board->board[0], 64 * sizeof (piece_t));
    for (i = count = 0; i < board->moves_count; i = ++count) {


        moveret = move(board, i);
        //printf("moveret %d", moveret);
        //print_board(&board->board[0]);
 

        /* move returns 1 on success */
        if (moveret == 1) {
            //score the move
            scores[i] = eval_curcuit(board->board, ai);
            undo_move(board, i);

            continue;
        }

        /* move returns -1 if stalemate, 0 if i > board->moves_count */
        if (!moveret)
            break;
        else if (moveret == -1)
            return -1;
        else
            printf("ERROR: moveret error\n");
    }

    //check if a port had constant output
    for (j = 0; j < ai->nr_ports; j++) {
        ai->separation_count[j] += (ai->separation[j][0] && ai->separation[j][1]);
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
        for (j = 0; j < 2; j++)
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
int do_best_move(AI_instance_t *ai, board_t * board, struct uci *uci_iface)
{
    int best_move;

    generate_all_moves(board);
    if (is_checkmate(board))
        return -1;
    if (is_stalemate(board) || (best_move = _get_best_move(ai, board)) == -1) {

        if (is_checkmate(board))
            return -1;

        return 0;
    }

    int ret = do_move(board, best_move);
    if (!ret)
        fprintf(stderr, "ret %d\n", ret);


    if (uci_iface)
        register_move_to_uci(&board->moves[best_move], uci_iface, board);
    swapturn(board);


    return 1;
}

//perform a random move return 0 if stalemate, -1 if check mate 1 of success
int do_random_move(board_t *board, struct uci *uci_iface)
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

    } while (do_move(board, rndmove) != 1);
    swapturn(board);

    if (uci_iface) {
        //printf("register move in random move \n");
        register_move_to_uci(&board->moves[rndmove], uci_iface, board);
    }

    return 1;
}

/* generate moves for a pice and make a move.
 * piece_type must be one of the values defined in enum moves_index. if EMPTY is
 * passed, a random piece is choosed */
int do_move_piece(board_t *board, enum moves_index piece_type,
        struct uci *uci_iface)
{
    int rnd, i, ret = 0;
    static void (*move_gen_callbacks[6])(board_t * board) = {
        generate_pawn_moves_only,
        generate_rook_moves_only,
        generate_knight_moves_only,
        generate_bishop_moves_only,
        generate_queen_moves_only,
        generate_king_moves_only,
    };

    if (piece_type == EMPTY) {
        // pick a random piece and generate moves for it. pick another piece if
        // the current one has no possible moves
        i = rnd = random_int_r(0, 5);
        do {
            i = (i + 1) % 6;
            move_gen_callbacks[i](board);
            if (board->moves_count > 0) {
                ret = do_random_move(board, uci_iface);
                if (ret == 1)
                    return 1;
            }
        } while (i != rnd);
        return ret;
    }

    if (piece_type >= 0 && piece_type < 6) {
        move_gen_callbacks[piece_type](board);
        // the stalemate (if EMPTY was passed) or no possible moves for piece the user specified
        if (board->moves_count <= 0)
            return 0;
        // will return a random move from the list of moves created in this function
        return do_random_move(board, uci_iface);
    }

    return -2; // user passed invalid value in piece_type
}

int do_move_random_piece(board_t *board, struct uci *uci_iface)
{
    return do_move_piece(board, EMPTY, uci_iface);
}

//perform a nonrandom move return 0 if stalemate, -1 if check mate 1 of success
int do_nonrandom_move(board_t *board, struct uci *uci_iface)
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
        rndmove = board->moves_count - 1;
    } while (!do_move(board, rndmove));

    if (uci_iface)
        register_move_to_uci(&board->moves[rndmove], uci_iface, board);

    swapturn(board);

    return 1;
}

void punish(AI_instance_t * ai)
{
    ai->nr_losses++;
    ai->nr_games_played++;
    int j;

    for (j = 0; j < ai->nr_ports; j++) {

        ai->state_separation_count[j] += (ai->state_separation[j][0] && ai->state_separation[j][1]);
    }

    bzero(&ai->state_separation[0][0], 2 * ai->nr_ports * sizeof (int));
}

void small_punish(AI_instance_t * ai)
{

    ai->nr_games_played++;
}

void small_reward(AI_instance_t *ai, int reward)
{

    ai->positive_reward += reward;
}

void reward(AI_instance_t * ai)
{
    ai->nr_wins += 1;
    ai->nr_games_played++;
    int j;

    for (j = 0; j < ai->nr_ports; j++) {

        ai->state_separation_count[j] += (ai->state_separation[j][0] && ai->state_separation[j][1]);
    }

    bzero(&ai->state_separation[0][0], 2 * ai->nr_ports * sizeof (int));

}

void draw(AI_instance_t *ai, board_t * board)
{
    //small_reward(ai, score_board(board));
    int j;

    for (j = 0; j < ai->nr_ports; j++) {

        ai->state_separation_count[j] += (ai->state_separation[j][0] && ai->state_separation[j][1]);
    }

    bzero(&ai->state_separation[0][0], 2 * ai->nr_ports * sizeof (int));

    ai->nr_games_played++;

}

void copy_ai(AI_instance_t *from_ai, AI_instance_t * to_ai)
{
    //copy a2 over to a1

    memcpy(&to_ai->brain[0][0], &from_ai->brain[0][0], to_ai->nr_ports * to_ai->nr_synapsis / 8);
    memcpy(&to_ai->port_type[0], &from_ai->port_type[0], to_ai->nr_ports * sizeof (int));
    memcpy(&to_ai->output_tag[0], &from_ai->output_tag[0], to_ai->nr_ports * sizeof (int));

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
    to_ai->low_port = from_ai->low_port;
    to_ai->high_port = from_ai->high_port;

}

void mutate_mutation_rates(AI_instance_t *ai)
{

    int max_val = 100;
    int min_val = 0;

    ai->output_rate -= random_int_r(0, 10);
    ai->output_rate += random_int_r(0, 10);
    ai->output_rate = keep_in_range(ai->output_rate, min_val, max_val);


    ai->r_output_rate -= random_int_r(0, 10);
    ai->r_output_rate += random_int_r(0, 10);
    ai->r_output_rate = keep_in_range(ai->r_output_rate, min_val, max_val);

    ai->unused_rate -= random_int_r(0, 10);
    ai->unused_rate += random_int_r(0, 10);
    ai->unused_rate = keep_in_range(ai->unused_rate, min_val, max_val);

    ai->state_separation_rate -= random_int_r(0, 10);
    ai->state_separation_rate += random_int_r(0, 10);
    ai->state_separation_rate = keep_in_range(ai->state_separation_rate, min_val, max_val);

    ai->separation_rate -= random_int_r(0, 10);
    ai->separation_rate += random_int_r(0, 10);
    ai->separation_rate = keep_in_range(ai->separation_rate, min_val, max_val);

    ai->state_separation_threshold -= random_int_r(0, 10);
    ai->state_separation_threshold += random_int_r(0, 10);
    ai->state_separation_threshold = keep_in_range(ai->state_separation_threshold, min_val, max_val);

    ai->separation_threshold -= random_int_r(0, 10);
    ai->separation_threshold += random_int_r(0, 10);
    ai->separation_threshold = keep_in_range(ai->separation_threshold, min_val, max_val);

    ai->port_type_rate -= random_int_r(0, 10);
    ai->port_type_rate += random_int_r(0, 10);
    ai->port_type_rate = keep_in_range(ai->port_type_rate, min_val, max_val);

    ai->zero_rate -= random_int_r(0, 10);
    ai->zero_rate += random_int_r(0, 10);
    ai->zero_rate = keep_in_range(ai->zero_rate, min_val, max_val);

    ai->one_rate -= random_int_r(0, 10);
    ai->one_rate += random_int_r(0, 10);
    ai->one_rate = keep_in_range(ai->one_rate, min_val, max_val);

    ai->output_exponent -= random_int_r(0, 1);
    ai->output_exponent += random_int_r(0, 1);
    ai->output_exponent = keep_in_range(ai->output_exponent, min_val, max_val);

}

void create_random_connections(AI_instance_t *a1, int nr_connections, int port, int pc)
{
    int i;
    if (pc || random_int_r(0, 1)) {
        for (i = 0; i < nr_connections && a1->low_port < port; i++) {
            int r = random_int_r(a1->low_port, port - 1);
            if (r > port - 1 || r < a1->low_port)
                printf("ERROR: illegal connection created: %d\n", r);
            SetBit(a1->brain[port], r);
        }

    } else {
        for (i = 0; i < nr_connections; i++)
            SetBit(a1->brain[port], random_int_r(a1->nr_ports, a1->nr_synapsis - 1));

    }

}

//check for constant ports and reset them
void reset_constant_ports(AI_instance_t *a1, AI_instance_t * a2)
{
    int j;
    for (j = a1->low_port; j <= a1->high_port; j++) {
        //ports that doesn't change in 1000 games is reset 
        if (((1000 * a2->separation_count[j]) / (1 + a2->nr_games_played)) == 0 &&
                ((1000 * a2->state_separation_count[j]) / (1 + a2->nr_games_played)) == 0) {
            if (random_int_r(MIN_VAL, MAX_VAL) > 90) {

                bzero(a1->brain[j], a1->nr_synapsis / 8);
                a1->port_type[j] = random_int_r(1, a1->nr_porttypes);
                a1->invert[j] = random_int_r(0, 1);


                create_random_connections(a1, 2, j, 0);

            }
        }
    }
}

//delete all connections for a port and create new ones
void randomize_ports(AI_instance_t *a1)
{
    int i;
    //completely randomize a port
    for (i = 0; i < 1; i++) {
        int r_port = random_int_r(a1->low_port, a1->high_port);
        a1->port_type[r_port] = random_int_r(1, a1->nr_porttypes);
        a1->invert[r_port] = random_int_r(0, 1);

        bzero(a1->brain[r_port], a1->nr_synapsis / 8);
        create_random_connections(a1, 2, r_port, 0);
    }
}

//find ports that are used by by output ports
void find_ports_in_use(AI_instance_t *a1)
{
    int i, j;
    int max_depth = 7;
    __m128i ad, bd;

    for (i = 0; i < (a1->nr_ports) / 32; i += 4) {

        bd = _mm_loadu_si128((__m128i *) (((int*) a1->brain[0]) + i));
        for (j = 0; j < a1->nr_ports; j++) {
            ad = _mm_loadu_si128((__m128i *) (((int*) a1->brain[j]) + i));
            bd = _mm_or_si128(ad, bd);
        }
        _mm_storeu_si128((__m128i *) (((int*) a1->used_port + i)), bd);
    }
    int k;
    for (k = 0; k < max_depth; k++) {
        for (i = 0; i < (a1->nr_ports) / 32; i += 4) {

            bd = _mm_loadu_si128((__m128i *) (((int*) a1->brain[0]) + i));
            for (j = 0; j < a1->nr_ports; j++) {
                if (TestBit(a1->used_port, j) || a1->output_tag[j]) {
                    ad = _mm_loadu_si128((__m128i *) (((int*) a1->brain[j]) + i));
                    bd = _mm_or_si128(ad, bd);
                }
            }
            _mm_storeu_si128((__m128i *) (((int*) a1->used_port + i)), bd);
        }
        //dump(a1->used_port, a1->nr_ports / 32);
    }
}

//move a port in ai with its connections, this should have no effect on AI output
void move_port(AI_instance_t *ai, int from, int to)
{
    // printf("from %d to %d\n", from, to);
    memcpy(&ai->brain[to][0], &ai->brain[from][0], ai->nr_synapsis / 8);
    ai->port_type[to] = ai->port_type[from];
    ai->output_tag[to] = ai->output_tag[from];
    ai->output_tag[from] = 0;
    if (TestBit(ai->used_port, from)) {
        SetBit(ai->used_port, to);
        ClearBit(ai->used_port, from);
    }
    int i;
    for (i = ai->low_port; i <= ai->high_port; i++) {
        //printf("hp: %d, i: %d, from: %d\n",ai->high_port, i, from);
        if (TestBit(ai->brain[i], from)) {
            SetBit(ai->brain[i], to);
            ClearBit(ai->brain[i], from);
        }
    }
}

void defrag(AI_instance_t *ai)
{
    int i;
    int j;
    for (i = ai->low_port; i <= ai->high_port;) {

        //is the port in use?
        if (!TestBit(ai->used_port, i) && !ai->output_tag[i]) {

            //if there are more ports in the lower half of brain than the upper half
            if (ai->nr_ports / 2 - ai->low_port >= ai->high_port - ai->nr_ports / 2) {
                for (j = i - 1; j >= ai->low_port; j--) {
                    move_port(ai, j, j + 1);
                }
                ai->low_port++;
                ai->low_port = keep_in_range(ai->low_port, 0, ai->nr_ports / 2);
                i++; //low port increased then so should i
            } else {
                for (j = i + 1; j <= ai->high_port; j++) {

                    move_port(ai, j, j - 1);
                }
                ai->high_port--;

            }
        } else //increase i if the port is in use
            i++;
    }
}

//brain in a a1 is replaced with brain from a2 plus a mutation
int mutate(AI_instance_t *a1, AI_instance_t * a2, int print, int print_stats)
{
    int i;
    int max_val = 100;
    int min_val = 0;
    copy_ai(a2, a1);
    
    //mutate_mutation_rates(a1);
    a1->low_port -= 1;
    a1->low_port = keep_in_range(a1->low_port, 0, a1->nr_ports - 1);
    create_random_connections(a1, 2, a1->low_port, 0);

    //chance to add a new port with random connections
    if (!random_int_r(0, 10)) {
        a1->high_port += 1;
        a1->high_port = keep_in_range(a1->high_port, 0, a1->nr_ports - 1);

        a1->output_tag[a1->high_port] = 1;
        a1->port_type[a1->high_port] = random_int_r(1, a1->nr_porttypes);
        create_random_connections(a1, 2, a1->high_port, 1);
    }
    //chance to set one port as an output port
    if (random_int_r(min_val, max_val) > 99) {
        a1->output_tag[random_int_r(a1->low_port, a1->high_port)] = 1;
    }


    //chance to remove the output tag
    if (random_int_r(min_val, max_val) > 20) {
        a1->output_tag[random_int_r(a1->low_port, a1->high_port)] = 0;
    }
    //printf("out_tag: %d\n", a1->output_tag[a1->low_port]);



    reset_constant_ports(a1, a2);

    //reset_low_entropy_ports(a1, a2);
    if (random_int_r(min_val, max_val) > 50)
        randomize_ports(a1);


    //remove random connections
    for (i = 0; i < 5; i++) {
        int r_port = random_int_r(a1->low_port, a1->high_port);
        int r_synaps = 0;
        r_synaps = random_int_r(0, a1->nr_synapsis - 1);

        ClearBit(a1->brain[r_port], r_synaps);
    }

    //create a new random connections
    if (random_int_r(min_val, max_val) > 50) {

        int r_port = random_int_r(a1->low_port, a1->high_port);
        create_random_connections(a1, 1, r_port, 0);
    }

    find_ports_in_use(a1);

    defrag(a1);


    //reset the new AI
    bzero(&a1->separation_count[0], a1->nr_ports * sizeof (int));
    bzero(&a1->state_separation_count[0], a1->nr_ports * sizeof (int));
    bzero(&a1->activation_count[0][0], 2 * a1->nr_ports * sizeof (int));
    bzero(&a1->state_separation[0][0], 2 * a1->nr_ports * sizeof (int));
    if (print_stats)
        fprintf(stderr, "score %f, %d w, %d games from score %f, %d w, %d games, one_rate: %d, zero_rate: %d, p_type_rate: %d, unused_rate: %d, o_rate: %d, r_o_rate: %d, separation_threshold: %d, state_separation_threshold: %d, separation_rate: %d, state_separation_rate: %d, exponent: %d, low_port: %d, high_port: %d\n",
            get_score(a1), a1->nr_wins, a1->nr_games_played, get_score(a2), a2->nr_wins, a2->nr_games_played, a2->one_rate, a2->zero_rate, a2->port_type_rate, a2->unused_rate, a2->output_rate, a2->r_output_rate, a2->separation_threshold, a2->state_separation_threshold, a2->separation_rate, a2->state_separation_rate, a2->output_exponent, a2->low_port, a2->high_port);
    clear_score(a1);

    return 1;
}

float get_score(AI_instance_t * ai)
{

    return ((float) (ai->nr_wins)) / ((float) ai->nr_games_played + 1); // + ai->positive_reward/((float)ai->nr_games_played+1);
    //return ((float)(ai->nr_wins))/((float)ai->nr_losses+1);
}

void clear_score(AI_instance_t * ai)
{
    ai->nr_losses = ai->nr_wins = ai->nr_games_played = ai->positive_reward = 0;
}
