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
extern  _get_best_move(AI_instance_t *ai, board_t *board);
#define SetBit(A,k)     ( A[(k/32)] |= (1 << (k%32)) )
#define ClearBit(A,k)   ( A[(k/32)] &= ~(1 << (k%32)) )            
#define TestBit(A,k)    ( A[(k/32)] & (1 << (k%32)) )
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
    board_t *board = new_board("P7/8/8/7q/7Q/8/8/p7 w - - 0 1");

    ai = ai_new(num_layers, features, feature_density);

    memset(ai->layers[0][0], 0xff, 256);
    ai->layers[1][0][0] = 0xffff;

   // ai->layers[0][0][0] = 0x1000;
    //ai->layers[0][0][1] = 0x0000;

    ret = score(ai, board->board);

    printf("board:\n");
    dump(board->board, 64*2*2);

    printf("layer 0:\n");
    dump(ai->layers[0][0], 64*2*2);

    printf("layer 1:\n");
    dump(ai->layers[1][0], 1*2);


    printf("score ret: %d\n", ret);

}


void do_best_move_test(void)
{
    int num_layers = 3, ret;
    int features[] = {10, 100, 1}, feature_density = 10;
    board_t *board = new_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");



  AI_instance_t *ai1;
    ai1 = ai_new(num_layers, features, feature_density);
 memset(ai1->layers[2][0], 0x0002, features[1]*2);
 srand(time(NULL));
    int i,j;
    for(i = 0; i < 128; i++){
        for(j = 0; j < features[0]; j++){
            int r = rand();
            //printf("r = %d",r);
            if(r%50)
                ai1->layers[0][j][i] = 0x0000;
        }
    }
    for(i = 0; i < 100; i++){
     do_best_move(ai1, board);
    }
}





void ai_test(void)
{
    int num_layers = 3, ret;
    int features[] = {10, 100, 1}, feature_density = 10;
    board_t *board = new_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");



  AI_instance_t *ai1;
    ai1 = ai_new(num_layers, features, feature_density);
 memset(ai1->layers[2][0], 0x0002, features[1]*2);
 srand(time(NULL));
    int i,j;
    for(i = 0; i < 128; i++){
        for(j = 0; j < features[0]; j++){
            int r = rand();
            //printf("r = %d",r);
            if(r%50)
                ai1->layers[0][j][i] = 0x0000;
        }
    }

  AI_instance_t *ai2;
    ai2 = ai_new(num_layers, features, feature_density);
 memset(ai2->layers[2][0], 0x0002, features[1]*2);
 srand(time(NULL));
   
    for(i = 0; i < 128; i++){
        for(j = 0; j < features[0]; j++){
            int r = rand();
            //printf("r = %d",r);
            if(r%50)
                ai2->layers[0][j][i] = 0x0000;
        }
    }
int iteration = 0;
while(1){
    printf("iteration: %d\n", iteration++);
    int ai1_won = 0;
    int ai2_won = 0;
    int game_pr_e = 100;
    int games = 0;
    while(games++ < game_pr_e){
    
    board_t *board = new_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    int max_moves = 100;
    int moves = 0;
    while(moves++ < max_moves){
        
        int ret = do_best_move(ai1, board);
        if(ret == 0){
       //     printf("STALEMATE\n");
            //print_board(board->board);
            break;
        }
        else if(ret == -1){
       //    printf("AI Lost\n");
            break;
        }
        //print_board(board->board);


        ret = do_random_move(board);
        if(ret == 0){
         //   printf("STALEMATE\n");
          //  print_board(board->board);
            break;
        }
        else if(ret == -1){
         //   printf("AI WON\n");
            ai1_won++;
            break;
        }
        //print_board(board->board);
     }
    }
    printf("ai1_won: %d\n", ai1_won);




games = 0;

while(games++ < game_pr_e){
    
    board_t *board = new_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    int max_moves = 100;
    int moves = 0;
    while(moves++ < max_moves){
        
        int ret = do_best_move(ai2, board);
        if(ret == 0){
           // printf("STALEMATE\n");
            //print_board(board->board);
            break;
        }
        else if(ret == -1){
           //printf("AI Lost\n");
            break;
        }
        //print_board(board->board);


        ret = do_random_move(board);
        if(ret == 0){
            //printf("STALEMATE\n");
          //  print_board(board->board);
            break;
        }
        else if(ret == -1){
           // printf("AI WON\n");
            ai2_won++;
            break;
        }
        //print_board(board->board);
     }
    }
    printf("ai2_won: %d\n", ai2_won);
   // dump(ai1->brain,ai1->nr_synapsis* ai1->nr_synapsis/32);
 //dump(ai1->brain,32);
 //        dump(ai2->brain,32);

if(ai1_won > ai2_won){
        mutate(ai2,ai2);
        printf("mutating ai2 from ai1\n");
    }
    else{
        mutate(ai1,ai2);
        printf("mutating ai1 from ai2\n");
    }
 //dump(ai1->brain,32);
 //dump(ai2->brain,32);

}



}

void mutate_test(){
int num_layers = 3, ret;
    int features[] = {1, 2, 1}, feature_density = 10;
    AI_instance_t *ai1, *ai2;
    board_t *board = new_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");

    ai1 = ai_new(num_layers, features, feature_density);
    ai2 = ai_new(num_layers, features, feature_density);

    printf("_______BEFORE______\n\n");
    printf("--AI1--\n");
    printf("layer 0:\n");
    dump(ai1->layers[0][0], 64*2*2);

    printf("layer 1:\n");
    dump(ai1->layers[1][0], 2*features[0]);
    dump(ai1->layers[1][1], 2*features[0]);

    printf("layer 2:\n");
    dump(ai1->layers[2][0], 2*features[1]);

    printf("\n--AI2--\n");
    printf("layer 0:\n");
    dump(ai2->layers[0][0], 64*2*2);

    printf("layer 1:\n");
    dump(ai2->layers[1][0], 2*features[0]);
    dump(ai2->layers[1][1], 2*features[0]);

    printf("layer 2:\n");
    dump(ai2->layers[2][0], 2*features[1]);


    mutate(ai1, ai2);
printf("\n\n_______AFTER_______\n\n");
    printf("--AI1--\n");
    printf("layer 0:\n");
    dump(ai1->layers[0][0], 64*2*2);

    printf("layer 1:\n");
    dump(ai1->layers[1][0], 2*features[0]);
    dump(ai1->layers[1][1], 2*features[0]);

    printf("layer 2:\n");
    dump(ai1->layers[2][0], 2*features[1]);

    printf("\n--AI2--\n");
    printf("layer 0:\n");
    dump(ai2->layers[0][0], 64*2*2);

    printf("layer 1:\n");
    dump(ai2->layers[1][0], 2*features[0]);
    dump(ai2->layers[1][1], 2*features[0]);

    printf("layer 2:\n");
    dump(ai2->layers[2][0], 2*features[1]);

}
void malloc_2d_test(void)
{
    int i, j;
    int **arr = malloc_2d(2, 100, sizeof(int));

    for (i = 0; i < 2; i++)
        for (j = 0; j < 100; j++)
            arr[i][j] = random_int_r(1, 100 * (i + 1));

    for (i = 0; i < 2; i++)
        for (j = 0; j < 100; j++)
            printf("[%d][%d] = %d\n", i, j, arr[i][j]);

    random_fill(&arr[0][0], 200 * sizeof(int));
    for (i = 0; i < 2; i++)
        for (j = 0; j < 100; j++)
            printf("[%d][%d] = %d\n", i, j, arr[i][j]);

    free(arr);
}


int nandscore_test(){
    int nr_ports = 64;
    int board_size = 64*2*2*8;
    int synapsis = nr_ports + board_size;

    int *V = (int *)malloc((( nr_ports)/32)*sizeof(int)); 
    int **M = malloc_2d(synapsis/32, synapsis,  4);
    int i;
    board_t *board = new_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");

    //int **board = (int *)malloc(64*2*2); 
    for (i = 0; i < nr_ports; i++)
            random_fill(&M[i][0], synapsis/8);

    printf("M[0][0]: %d\n", M[0][0]);
    //printf("M[0]: %d\n", TestBit(M[0],0));

 

    printf("ret from eval: %d\n", eval_curcuit(V, M, nr_ports, board->board, board_size));
    printf("V : %x\n ", &V);
    for(i = 0; i < nr_ports; i++){

        if(TestBit(V,i))
            printf("1");
        else
            printf("0");    
    }
}

int main(int argc, char *argv[])
{
    //multiply_test();
    //score_test();
    //malloc_2d_test();
    //do_best_move_test();
   //mutate_test();
   ai_test();
 //bitarray_test();
//nandscore_test();   
 return 0;
}
