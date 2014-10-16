#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

#include "common.h"
#include "board.h"
#include "AI.h"
#include "board.h"

extern int8_t multiply(piece_t *features, piece_t *board, int n);
extern int8_t score(AI_instance_t *ai, piece_t *board);
extern  _get_best_move(AI_instance_t *ai, board_t *board);

unsigned long now(void)
{
    unsigned long ms;
    time_t s;
    struct timespec spec;

    clock_gettime(CLOCK_REALTIME, &spec);

    s  = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6); // convert nanoseconds to milliseconds

    return s * 1000 + ms;
}

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
        int game_pr_e = 1000;
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

        if(ai1_won > ai2_won){
            mutate(ai2,ai2);
            printf("mutating ai2 from ai1\n");
        }
        else{
            mutate(ai1,ai2);
            printf("mutating ai1 from ai2\n");
        }

    }



}

void mutate_test()
{
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
    int **arr = (int **)malloc_2d(100, 2, sizeof(int));

    for (i = 0; i < 2; i++)
        for (j = 0; j < 100; j++)
            arr[i][j] = random_int_r(1, 100 * (i + 1));

    random_fill(&arr[0][0], 200 * sizeof(int));

    free(arr);

    printf("%s suceeded\n", __FUNCTION__);
}

struct game_struct {
    int nr_games, thread_id;
    long checkmates, stalemates, timeouts;
};

void *moves_test(void *arg)
{
    int i, j, ret;
    board_t *board;
    long checkmate = 0, stalemate = 0, timeout = 0;
    struct game_struct *game = (struct game_struct *)arg;

    for (j = 0; j < game->nr_games; j++) {
        board = new_board(NULL);
        //board = new_board("Q6k/5K2/8/8/8/8/8/8 b - - 0 1");
        //board = new_board("Q6k/8/8/8/8/8/K7/6R1 b - - 0 1");

        for (i = 0; i < 100; i++) {
            generate_all_moves(board);
            ret = do_random_move(board);

#ifdef DEBUG
            print_board(board->board);
            bb_print(board->white_pieces.apieces);
            printf("thread %d: --- %d %s ---\n", game->thread_id, i, board->turn == WHITE ? "white" : "black");
            getchar();
#endif

            if (ret == 0) {
                debug_print("(%d) stalemate\n", game->thread_id);
                ++stalemate;
                break;
            } else if (ret == -1) {
                debug_print("(%d) checkmate\n", game->thread_id);
                ++checkmate;
                break;
            }
        }

        if (ret > 0)
            ++timeout;

        free(board);
    }

    game->checkmates = checkmate;
    game->stalemates = stalemate;
    game->timeouts = timeout;
}

void spawn_n_games(int n, int rounds)
{
    pthread_t threads[n - 1];
    int i, num_layers = 3, ret;
    int checkmate, stalemate, timeout;
    struct game_struct games[n];

    for (i = 0; i < n; i++) {
        games[i].nr_games = rounds;
        games[i].thread_id = i + 1;

        if (i == n - 1)
            break;
        pthread_create(&threads[i], NULL, moves_test, &games[i]);
    }

    moves_test(&games[i]);
    checkmate = games[i].checkmates;
    stalemate = games[i].stalemates;
    timeout = games[i].timeouts;

    for (i = 0; i < n - 1; i++) {
        pthread_join(threads[i], NULL);

        checkmate += games[i].checkmates;
        stalemate += games[i].stalemates;
        timeout += games[i].timeouts;
    }

    printf("%d checkmates\n", checkmate);
    printf("%d stalemates\n", stalemate);
    printf("%d timeouts\n", timeout);
}

int main(int argc, char *argv[])
{
    unsigned long start, end;
    double diff;
    int i, rounds, threads, count;
    //multiply_test();
    //score_test();
    //malloc_2d_test();
    //do_best_move_test();
    //mutate_test();
    //ai_test();

    init_magicmoves();

    start = now();
    rounds = argc > 1 ? atoi(argv[1]) : 2000;
    threads = argc > 2 ? atoi(argv[2]) : 1;
    spawn_n_games(threads, rounds);
    end = now();

    diff = end - start;

    count = rounds * threads;
    printf("%d games played in %.0f ms (%.1f games pr. second, w/ %d threads)\n",
            count, diff, (double)count / (diff / 1000), threads);

    return 0;
}
