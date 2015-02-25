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
    int ***brain_a; 
    int ***brain_b;       
    int nr_ports;
    int nr_synapsis;
    int board_size;
    int mutation_rate;
    int nr_brain_parts;
    int **port_type;
    int **mutationrate;
    int nr_porttypes;
    int ***activation_count;
    int ***correlations;
    int **invert;
    int nr_outputs;
    int ***separation;
    int **separation_count;
    int ***state_separation;
    int **state_separation_count;
    int **output_tag;
    int zero_rate;
    int one_rate;
    int port_type_rate;
    int unused_rate; 
    int output_rate;
    int r_output_rate;
    int separation_rate;
    int state_separation_rate;
    int separation_threshold;
    int state_separation_threshold;
    int *used_port;

} __attribute__((packed)) AI_instance_t;

extern AI_instance_t *ai_new(int mutation_rate, int brain_size, int nr_ports);
extern AI_instance_t *load_ai(char *file, int mutation_rate);
extern int dump_ai(char *file, AI_instance_t *ai);
extern void ai_free(AI_instance_t *ai);
extern int _get_best_move(AI_instance_t *ai, board_t * board);
extern int do_best_move(AI_instance_t *ai, board_t *board);
extern void punish(AI_instance_t *ai);
extern void reward(AI_instance_t *ai);
extern void draw(AI_instance_t *ai, board_t * board);
extern float get_score(AI_instance_t *ai);
extern int mutate(AI_instance_t *a1, AI_instance_t *a2, int print);
extern void clear_score(AI_instance_t *ai);
extern int do_nonrandom_move(board_t *board);
extern int do_random_move(board_t *board);
extern int do_pseudo_random_move(board_t *board);


extern int score(AI_instance_t *ai, piece_t *board);
extern int eval_curcuit(int *V, int **M,  int nr_ports, piece_t *board, int board_size, int* port_type, int **brain_a, int **brain_b, int **activation_count,int **state_separation, int *invert, int nr_outputs, int*output_tag, int *used_port);

#endif
