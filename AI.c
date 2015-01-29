#include "smmintrin.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "common.h"
#include "AI.h"

#define MAGIC_LENGTH 6
static unsigned char ai_mem_magic[] = "\x01\x02\x03\x04\x05\x06";

AI_instance_t *ai_new(int mutation_rate, int brain_size) {
    int i, j, k;
    AI_instance_t *ret;

    ret = calloc(1, sizeof (struct AI_instance));
    if (ret == NULL) {
        perror("malloc");
        return NULL;
    }

    ret->nr_ports = 512;
    ret->board_size = 64 * 2 * 2 * 8;
    ret->nr_synapsis = ret->nr_ports + ret->board_size;
    ret->nr_brain_parts = brain_size;

    ret->brain = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->brain_a = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->brain_b = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    memset(ret->brain_b[0][0], 0xff, ret->nr_synapsis / (sizeof (int) * 8) *
            ret->nr_ports * ret->nr_brain_parts * sizeof (int));

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
    printf("ret->brain %d\n", ret->nr_synapsis);
    _dump(ret->brain[0][ret->nr_ports - 1], ret->nr_synapsis / 8);

    ret->port_type = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    for (i = 0; i < ret->nr_brain_parts; i++) {
        for (j = 0; j < ret->nr_ports; j++) {
            ret->port_type[i][j] = random_int_r(1, 4);
        }
    }
    ret->nr_porttypes = 4;
    ret->mutationrate = (int**) malloc_2d(4, 2, sizeof (int));

    for (i = 0; i < 1; i++) {
        for (j = 0; j < ret->nr_porttypes; j++) {
            ret->mutationrate[i][j] = 10;
        }
    }

    for (i = 1; i < 2; i++) {
        for (j = 0; j < ret->nr_porttypes; j++) {
            ret->mutationrate[i][j] = 100;
        }
    }
    ret->activation_count = (int***) malloc_3d(2, ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    ret->move_nr = 0;
    ret->nr_wins = ret->nr_losses = ret->nr_games_played = 0;
    ret->generation = 0;
    ret->mutation_rate = mutation_rate;

    return ret;
}

AI_instance_t * copy_ai(AI_instance_t * ai) {
    AI_instance_t *ret = calloc(1, sizeof (AI_instance_t));

    memcpy(ret, ai, sizeof (AI_instance_t));
    ret->brain = (int ***) memdup_3d((void ***) ai->brain);

    clear_score(ret);
    return ret;
}

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
    fwrite(&ai->port_type[0][0], 1, ai->nr_ports * sizeof (int)*ai->nr_brain_parts, out);
    fwrite(&ai->mutationrate[0][0], 1, ai->nr_porttypes * 2 * sizeof (int), out);

    fclose(out);

    return 1;
}

AI_instance_t * load_ai(char *file, int mutation_rate) {
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

    fread(magic, 1, MAGIC_LENGTH, in);
    if (memcmp(magic, ai_mem_magic, MAGIC_LENGTH)) {
        fprintf(stderr, "%s is not a valid AI dump\n", file);
        free(ret);
        fclose(in);
        return NULL;
    }

    fread(ret, 1, sizeof (AI_instance_t), in);
    ret->brain_a = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    ret->brain = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    ret->brain_b = (int ***) malloc_3d(ret->nr_synapsis / (sizeof (int) * 8),
            ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    brain_size = (ret->nr_synapsis / (sizeof (int) * 8)) *
            ret->nr_ports * sizeof (int)*ret->nr_brain_parts;
    fread(&ret->brain[0][0][0], 1, brain_size, in);

    ret->port_type = (int**) malloc_2d(ret->nr_ports, ret->nr_brain_parts, sizeof (int));
    fread(&ret->port_type[0][0], 1, ret->nr_ports * sizeof (int)*ret->nr_brain_parts, in);

    ret->mutationrate = (int**) malloc_2d(ret->nr_porttypes, 2, sizeof (int));
    printf("nr_porttypes: %d\n", ret->nr_porttypes);
    fread(&ret->mutationrate[0][0], 1, ret->nr_porttypes * 2 * sizeof (int), in);


    ret->mutation_rate = mutation_rate;

    ret->activation_count = (int***) malloc_3d(2, ret->nr_ports, ret->nr_brain_parts, sizeof (int));

    printf("loading done \n", ret->nr_porttypes);



    fclose(in);

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

        ad = _mm_loadu_si128((__m128i *) (((int*) a) + i));
        bd = _mm_loadu_si128((__m128i *) (((int*) b) + i));
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

        ad = _mm_loadu_si128((__m128i *) (((int*) board) + i));
        bd = _mm_loadu_si128((__m128i *) ((int*) b + (i + (nr_ports / 32))));
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

int nand(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b) {
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

        ad = _mm_loadu_si128((__m128i *) (((int*) a) + i));
        bd = _mm_loadu_si128((__m128i *) (((int*) b) + i));
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

        ad = _mm_loadu_si128((__m128i *) (((int*) board) + i));
        bd = _mm_loadu_si128((__m128i *) ((int*) b + (i + (nr_ports / 32))));
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

        ad = _mm_loadu_si128((__m128i *) (((int*) a) + i));
        bd = _mm_loadu_si128((__m128i *) (((int*) b) + i));
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

        ad = _mm_loadu_si128((__m128i *) (((int*) board) + i));
        bd = _mm_loadu_si128((__m128i *) ((int*) b + (i + (nr_ports / 32))));
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

int or(int *a, int *b, int nr_ports, piece_t *board, int board_size, int * brain_a, int *brain_b) {
    __m128i ad, bd, cd;

    int i;
    for (i = 0; i < (nr_ports) / 32; i += 4) {
        ad = _mm_loadu_si128((__m128i *) (((int*) a) + i));
        bd = _mm_loadu_si128((__m128i *) (((int*) b) + i));
        cd = _mm_loadu_si128((__m128i *) (((int*) brain_a) + i));
        cd = _mm_or_si128(_mm_and_si128(ad, bd), cd);
        _mm_storeu_si128((__m128i *) (((int*) brain_a) + i), cd);

        ad = _mm_loadu_si128((__m128i *) (((int*) a) + i));
        bd = _mm_loadu_si128((__m128i *) (((int*) b) + i));
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

        ad = _mm_loadu_si128((__m128i *) (((int*) board) + i));
        bd = _mm_loadu_si128((__m128i *) ((int*) b + (i + (nr_ports / 32))));
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

int eval_curcuit(int *V, int **M, int nr_ports, piece_t *board, int board_size, int* port_type, int **brain_a, int **brain_b, int **activation_count) {

    int i;
    for (i = 0; i < nr_ports; i++) {
        // if(nand(V, M[i], nr_ports, board, board_size) != nand_validation(V, M[i], nr_ports, board, board_size))
        //     printf("NAND and NAND validation did not return same value\n");
        if (0 && i == nr_ports - 1) {
            printf("last port\n");
            _dump(M[i], nr_ports / 16);
            _dump(V, nr_ports / 16);
        }
        // if (i == 0)
        //     SetBit(V, i);
        if (port_type[i] == 1) {
            if (nand(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i]))
                SetBit(V, i);
        } else if (port_type[i] == 2) {
            if (or(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i]))
                SetBit(V, i);
        } else if (port_type[i] == 3) {
            if (nor(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i]))
                SetBit(V, i);
        } else if (port_type[i] == 4) {
            if (and(V, M[i], nr_ports, board, board_size, brain_a[i], brain_b[i]))
                SetBit(V, i);
        } else
            printf("ERROR: port type error(%d)", port_type[i]);
    }
    //printf("port values: \n");
    for (i = 0; i < nr_ports; i++) {
        //   printf("%d", !!!TestBit(V,i));
        activation_count[i][!!!TestBit(V, i)] += 1;
    }
    //printf("\n");
    //printf("eval port type: %d\n", port_type[nr_ports-1]);
    //_dump(brain_a[nr_ports-1], nr_ports/16);
    return !!!TestBit(V, (nr_ports - 1)) + !!!TestBit(V, (nr_ports - 2)) + !!!TestBit(V, (nr_ports - 3))
            + !!!TestBit(V, (nr_ports - 4)) + !!!TestBit(V, (nr_ports - 5)) + !!!TestBit(V, (nr_ports - 6))
            + !!!TestBit(V, (nr_ports - 7)) + !!!TestBit(V, (nr_ports - 8)) + !!!TestBit(V, (nr_ports - 9));
}

int score(AI_instance_t *ai, piece_t * board) {
    int V[(ai->nr_ports) / 32];

    int score_sum = 0;
    int i;
    for (i = 0; i < ai->nr_brain_parts; i++) {
        bzero(V, sizeof (V));
        score_sum += eval_curcuit(V, ai->brain[i], ai->nr_ports, board, ai->board_size, ai->port_type[i], ai->brain_a[i], ai->brain_b[i], ai->activation_count[i]);
    }
    //if(score_sum)
    //    printf("score_sum:%d", score_sum);
    return score_sum;
}

static int _get_best_move(AI_instance_t *ai, board_t * board) {
    int i, count, moveret;
    float cumdist[board->moves_count], fcount, x;
    int scores[board->moves_count];

    memcpy(&board->board[64], &board->board[0], 64 * sizeof (piece_t));
    for (i = count = 0; i < board->moves_count; i = ++count) {
        moveret = move(board, i);
        // printf("moveret: %d\n", moveret);
        /* move returns 1 on success */
        if (moveret == 1) {
            scores[i] = score(ai, board->board);
            //printf("score: %d\n", scores[i]);
            undo_move(board, i);
            //  if (scores[i] == 9)
            //      break;

            continue;
        }

        /* move returns -1 if stalemate, 0 if i > board->moves_count */
        if (!moveret)
            break;
        else if (moveret == -1)
            return -1;
    }

    fcount = 0;
    /*  int best_i = 0;
      int best_val = 0;
      for (i = 0; i < board->moves_count; i++) {
          if (best_val == 9)
              break;
          if (scores[i] > best_val) {
              best_val = scores[i];
              best_i = i;
          }
      }
      return best_i;
     */
    for (i = 0; i < board->moves_count; i++) {
        fcount += (scores[i] * scores[i]);
        cumdist[i] = fcount;
    }
    x = random_float() * cumdist[board->moves_count - 1];
    if (bisect(cumdist, x, board->moves_count) >= board->moves_count)
        printf("INVALID MOVE RETURNED\n");
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

    int ret = do_move(board, best_move);
    if (!ret)
        printf("ret %d\n", ret);
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
}

void draw(AI_instance_t *ai, board_t * board) {
    //small_reward(ai, score_board(board));

    ai->nr_games_played++;

}
//brain in a a1 is replaced with brain from a2 pluss a mutation

int mutate(AI_instance_t *a1, AI_instance_t * a2) {
    int i, j;

    memcpy(&a1->brain[0][0][0], &a2->brain[0][0][0], a1->nr_brain_parts * a1->nr_ports * a1->nr_synapsis / 8);
    memcpy(&a1->port_type[0][0], &a2->port_type[0][0], a1->nr_ports * a1->nr_brain_parts * sizeof (int));
    memcpy(&a1->mutationrate[0][0], &a2->mutationrate[0][0], 4 * 2 * sizeof (int));
    //memcpy(&a1->activation_count[0][0][0], &a2->activation_count[0][0][0], 2 * a1->nr_ports * a1->nr_brain_parts * sizeof (int));

    __m128i ad, bd, cd, dd;
    //printf("brain_a \n");
    //_dump(a2->brain_a[0][a1->nr_ports-1], a1->nr_synapsis / 8);
    printf("a2->brain\n");
    printf("%d, %d, %d, %d\n", a2->mutationrate[0][0], a2->mutationrate[0][1], a2->mutationrate[0][2], a2->mutationrate[0][3]);
    printf("%d, %d, %d, %d\n", a2->mutationrate[1][0], a2->mutationrate[1][1], a2->mutationrate[1][2], a2->mutationrate[1][3]);
    printf("mutation_rate: %d\n", a2->mutation_rate);
    a1->mutation_rate = a2->mutation_rate;
    
    //only transfer connections without constant value
    for (i = 0; i < (a1->nr_brain_parts * a1->nr_ports * a1->nr_synapsis) / 32; i += 4) {
        ad = _mm_loadu_si128((__m128i *) (((int*) a2->brain[0][0]) + i));
        bd = _mm_loadu_si128((__m128i *) ((int*) a2->brain_a[0][0] + (i)));
        dd = _mm_loadu_si128((__m128i *) ((int*) a2->brain_b[0][0] + (i)));

        cd = _mm_andnot_si128(dd, _mm_and_si128(ad, bd));
        _mm_storeu_si128((__m128i *) (((int*) a1->brain[0][0]) + i), cd);
    }

    //mutate mutation rate for the various type of ports
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 4; j++) {
            a1->mutationrate[i][j] -= random_int_r(0, 5);
            a1->mutationrate[i][j] += random_int_r(0, 5);
            if (a1->mutationrate[i][j] < 0)
                a1->mutationrate[i][j] = 0;
        }
    }
    
    //mutate the overall port mutation rate
    a1->mutation_rate -= random_int_r(0, 5);
    a1->mutation_rate += random_int_r(0, 5);
    if (a1->mutation_rate < 1)
        a1->mutation_rate = 1;
    
    //check for constant ports and reset them
    for (i = 0; i < a1->nr_brain_parts; i++) {
        for (j = 0; j < a1->nr_ports; j++) {
            float ratio = (float) (a2->activation_count[i][j][0] + 1) / ((float) a2->activation_count[i][j][1] + 1);
            if (ratio < 0.00001 || ratio > 100000) {
                printf("ratio: %.2f r_port: %d\n", ratio, j);
                bzero(a1->brain[i][j], a1->nr_synapsis / 8);
                a1->port_type[i][j] = random_int_r(1, a1->nr_porttypes);
                int r_brain = i;
                int r_port = j;
                int k;
                for (k = 0; k < a1->mutationrate[0][a1->port_type[r_brain][r_port]]; k++) {
                    int r_synaps = random_int_r(0, a1->nr_synapsis - 1);
                    if (r_port >= a1->nr_ports - 1 - 9 && r_synaps > a1->nr_ports - 1 - 9)
                        ClearBit(a1->brain[r_brain][r_port], r_synaps);
                    else
                        SetBit(a1->brain[r_brain][r_port], r_synaps);
                }
                for (k = 0; k < a1->mutationrate[1][a1->port_type[r_brain][r_port]]; k++) {
                    int r_synaps = random_int_r(0, a1->nr_synapsis - 1);
                    ClearBit(a1->brain[r_brain][r_port], r_synaps);
                }
            }
        }
    }
    
    //mutate ports
    for (i = 0; i < a1->mutation_rate; i++) {
        int r_brain = random_int_r(0, a1->nr_brain_parts - 1);
        int r_port = random_int_r(0, a1->nr_ports - 1);
        a1->port_type[r_brain][r_port] = random_int_r(1, a1->nr_porttypes);

        for (j = 0; j < a1->mutationrate[0][a1->port_type[r_brain][r_port]]; j++) {
            int r_synaps = random_int_r(0, a1->nr_synapsis - 1);
            if (r_port >= a1->nr_ports - 1 - 9 && r_synaps > a1->nr_ports - 1 - 9)
                ClearBit(a1->brain[r_brain][r_port], r_synaps);
            else
                SetBit(a1->brain[r_brain][r_port], r_synaps);
        }
        for (j = 0; j < a1->mutationrate[1][a1->port_type[r_brain][r_port]]; j++) {
            int r_synaps = random_int_r(0, a1->nr_synapsis - 1);
            ClearBit(a1->brain[r_brain][r_port], r_synaps);
        }
    }
    
    //reset the new AI
    bzero(a1->brain_a[0][0], a1->nr_brain_parts * a1->nr_ports * a1->nr_synapsis / 8);
    bzero(a1->activation_count[0][0], 2 * a1->nr_ports * a1->nr_brain_parts * sizeof (int));
    memset(a1->brain_b[0][0], 0xff, a1->nr_synapsis / (sizeof (int) * 8) *
            a1->nr_ports * a1->nr_brain_parts * sizeof (int));
    clear_score(a1);
   
    return 1;
}

int mutate2(AI_instance_t *a1, AI_instance_t * a2) {
    int i, j;
    unsigned r1, r2, r3;
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
    return ((float) (ai->nr_wins)) / ((float) ai->nr_games_played + 1); // + ai->positive_reward/((float)ai->nr_games_played+1);
    //return ((float)(ai->nr_wins))/((float)ai->nr_losses+1);
}

void clear_score(AI_instance_t * ai) {
    ai->nr_losses = ai->nr_wins = ai->nr_games_played = ai->positive_reward = 0;
}
