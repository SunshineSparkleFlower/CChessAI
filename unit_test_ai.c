#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include "common.h"
#include "board.h"
#include "AI.h"

#include "fastrules.h"
#include "rules.h"
#include "board.h"


extern int8_t multiply(piece_t *features, piece_t *board, int n);
extern int8_t score(AI_instance_t *ai, piece_t *board);

void multiply_test(void)
{
    int num_layers = 2, ret;
    int features[] = {1, 1}, feature_density = 10;
    AI_instance_t *ai;
    board_t *board = new_board("P7/8/8/8/8/8/8/8 w - - 0 1");

    ai = ai_new(num_layers, features, feature_density);

    printf("board:\n");
    dump(board->board, 64*2*2);

    memset(ai->layers[0][0], 0xff, 256);
    memset(ai->layers[1][0], 0xff, 256);

    ai->layers[0][0][0] = 0x1000;
    ai->layers[0][0][1] = 0x3000;

    printf("layer 0:\n");
    dump(ai->layers[0][0], 64*2*2);

    ret = multiply(ai->layers[0][0], board->board, 64*2);

    printf("multiply ret: %d\n", ret);
}

void score_test(void)
{
    int num_layers = 2, ret;
    int features[] = {1, 1}, feature_density = 10;
    AI_instance_t *ai;
    board_t *board = new_board("P7/8/8/8/8/8/8/8 w - - 0 1");

    ai = ai_new(num_layers, features, feature_density);

    memset(ai->layers[0][0], 0xff, 256);
    ai->layers[1][0][0] = 0xff05;

    ai->layers[0][0][0] = 0x1000;
    ai->layers[0][0][1] = 0x0000;

    ret = score(ai, board->board);

    printf("board:\n");
    dump(board->board, 64*2*2);

    printf("layer 0:\n");
    dump(ai->layers[0][0], 64*2*2);

    printf("layer 1:\n");
    dump(ai->layers[1][0], 1*2);


    printf("score ret: %d\n", ret);

}

int main(int argc, char *argv[])
{
    //multiply_test();
    score_test();

    return 0;
}
