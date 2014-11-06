#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>
#include <ctype.h>
#include <string.h>

#include "common.h"
#include "board.h"
#include "AI.h"
#include "threadpool.h"

struct game_struct {
    AI_instance_t *ai;
    int games_to_play, max_moves;
    int (*do_a_move)(board_t *);
    char *fen;

    int game_id;
};

int nr_threads = 2, nr_jobs = 2, i, best, iteration;
struct game_struct *games = NULL;
struct job *jobs = NULL;

void print_ai_stats(int tid, AI_instance_t *ai, int ite, int rndwins)
{
    printf("thread %d: iteration %d\n", tid, ite);
    printf("thread %d: aiw nr wins: %d\n", tid, ai->nr_wins);
    printf("thread %d: random wins: %d\n", tid, rndwins);
    printf("thread %d: memory attributes:\n", tid);
    printf("    P and R\n");
    printf("thread %d: generation: %d\n", tid, ai->generation);
}

void play_chess(void *arg)
{
    struct game_struct *game = (struct game_struct *)arg;
    int games_to_play, nr_games, max_moves, moves, ret = -1;
    int (*do_a_move)(board_t *);
    AI_instance_t *ai;
    board_t *board;

    ai = game->ai;
    games_to_play = game->games_to_play;
    max_moves = game->max_moves;
    do_a_move = game->do_a_move;

    //printf("starting game %d\n", game->game_id);

    for (nr_games = 0; nr_games < 50; nr_games++) {
        board = new_board(game->fen);
        //board_t *board = new_board("rnbqkbnr/qqqqqqqq/8/8/8/8/qqqqqqqq/qqqqKqqq w - - 0 1");

        for (moves = 0; moves < max_moves; moves++) {
            ret = do_best_move(ai, board);
            if(ret == 0) {
                break;
            } else if (ret == -1) {
                punish(ai);
                break;
            }

            ret = do_a_move(board);
            if(ret == 0) {
                break;
            } else if(ret == -1) {
                reward(ai);
                break;
            }
        }
      if(ret == 0 || moves == max_moves)
          draw(ai, board);
      //  if (ret >= 0){
      //          small_reward(ai,score_board(board->board));            
      //  }
        free_board(board);
    }
}

int get_best_ai(struct game_struct *g, int n, int lim)
{
    int i, best = lim ? 0 : 1;
    float best_val = -1000000.0;
    for (i = 0; i < n; i++) {
        if (get_score(g[i].ai) > best_val && lim != i && g[i].games_to_play <= g[i].ai->nr_games_played){
            best = i;
            best_val = get_score(g[i].ai);
        }
    }

    return best;
}

void sighandler(int sig)
{
    char buffer[512], file[128];
    int n = 1, i, best;
    struct timeval tv;
    struct tm *tm;

    gettimeofday(&tv, NULL);
    tm = localtime(&tv.tv_sec);
    strftime(file, 64, "ai_save-%d%m%y-%H%M", tm);

    fprintf(stderr, "Are you sure you want to quit? (Y/n): [default: n] ");
    fgets(buffer, sizeof(buffer), stdin);
    if (tolower(buffer[0]) == 'c') //exit without any further questions
        exit(0);

    if (tolower(buffer[0]) != 'y')
        return;
    
    
    fprintf(stderr, "Save N best AIs to file? (Y/n): [default: y] ");
    fgets(buffer, sizeof(buffer), stdin);
    if (tolower(buffer[0]) == 'n')
        n = 0;

    if (n) {
        fprintf(stderr, "N (0-%d): [default: %d] ", nr_jobs, n);
        fgets(buffer, sizeof(buffer), stdin);
        buffer[127] = 0;
        if (buffer[0] != '\n')
            n = atoi(buffer);
        if (n > nr_jobs) {
            fprintf(stderr, "N > %d, setting N to %d\n", nr_jobs, nr_jobs);
            n = nr_jobs;
        }

        fprintf(stderr, "File prefix (<= 64 chars): [default: %s] ", file);
        fgets(buffer, 64, stdin);
        if (buffer[0] != '\n') {
            strncpy(file, buffer, 64);
            if (file[strlen(file) - 1] == '\n')
                file[strlen(file) - 1] = 0;
            file[64] = 0;
        }
    }

    for (i = 0; i < n; i++) {
        best = get_best_ai(games, nr_jobs, -1);
        sprintf(buffer, "%s-%d.aidump", file, i + 1);
        fprintf(stderr, "saving ai to file '%s'\n", buffer);
        dump_ai(buffer, games[best].ai);
        clear_score(games[best].ai);
    }

    exit(0);
}

void natural_selection(void)
{
    int ai1 = random_int_r(0,nr_jobs-1);
    int ai2 = random_int_r(0,nr_jobs-1);
    float score_ai1  = get_score(games[ai1].ai);
    float score_ai2 =  get_score(games[ai2].ai);
    if(score_ai1 > score_ai2){
        printf("mutating ai%d (score %f, %d wins, %d losses)",
                ai2, get_score(games[ai2].ai),
                games[ai2].ai->nr_wins, games[ai2].ai->nr_losses);
        printf(" from ai%d (score %f, %d wins, %d losses)\n",
                ai1, get_score(games[ai1].ai),
                games[ai1].ai->nr_wins, games[ai1].ai->nr_losses);

        mutate(games[ai2].ai, games[ai1].ai);
    }
    else if(score_ai2 > score_ai1){
        printf("mutating ai%d (score %f, %d wins, %d losses)",
                ai1, get_score(games[ai1].ai),
                games[ai1].ai->nr_wins, games[ai1].ai->nr_losses);
        printf(" from ai%d (score %f, %d wins, %d losses)\n",
                ai2, get_score(games[ai2].ai),
                games[ai2].ai->nr_wins, games[ai2].ai->nr_losses);

        mutate(games[ai1].ai, games[ai2].ai);   
    }
    
   
 //   best = get_best_ai(games, nr_jobs, -1);
    
 // printf("BEST: ai%d (score %f, %d wins, %d losses, wlr: %f\n)",
//                best, get_score(games[best].ai),
 //               games[best].ai->nr_wins, games[best].ai->nr_losses, games[best].ai->nr_wins/(float)games[best].ai->nr_losses);
  
}


int main(int argc, char *argv[])
{
    char *ai_file = NULL;

  int j;
  int mutation_rate = 1000;
  int games_to_play = 100;
  int c;
  int max_iterations = 100;
  int brain_size = 3;
  int selection_function = 0;
  int nr_selections = 1;
  opterr = 0;
  while ((c = getopt (argc, argv, "t:j:f:m:g:i:b:s:r:")) != -1)
    switch (c)
      {
      case 't':
        nr_threads = atoi(optarg);
        break;
      case 'j':
        nr_jobs = atoi(optarg);
        break;
      case 'f':
        ai_file = optarg;
        break;
      case 'm':
        mutation_rate = atoi(optarg);
        break;
      case 'g':
        games_to_play = atoi(optarg);
        break;
      case 'i':
        max_iterations = atoi(optarg);
        break;
     case 'b':
        brain_size = atoi(optarg);
        break;
     case 's':
       selection_function = atoi(optarg);
        break;
     case 'r':
        nr_selections = atoi(optarg);
        break;
      case '?':
        if (optopt == 'f')
          fprintf (stderr, "Option -%c requires an file name.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,
                   "Unknown option character `\\x%x'.\n",
                   optopt);
        printf("USAGE: %s -t <nr threads> -j <nr jobs> -f <ai_filename> -m <mutation rate>\n", argv[0]);

        return 1;
      default:
        abort ();
      }


    srand(100);

    if (nr_threads == 0 || nr_jobs == 0) {
        printf("threads or jobs cannot be 0\n");
        printf("USAGE: %s -t <nr threads> -j <nr jobs> -f <ai_filename> -m <mutation rate>\n", argv[0]);
        return 1;
    }

    init_threadpool(nr_threads);
    init_magicmoves();

    signal(SIGINT, sighandler);

    jobs = malloc(nr_jobs * sizeof(struct job));
    games = malloc(nr_jobs * sizeof(struct game_struct));

    for (i = 0; i < nr_jobs; i++) {
        if (ai_file) {
            games[i].ai = load_ai(ai_file, mutation_rate);
            clear_score(games[i].ai);
        } else
            games[i].ai = ai_new(mutation_rate, brain_size); // mutation rate = 5000
        if (games[i].ai == NULL) {
            perror("ai creation");
            exit(1);
        }
        games[i].games_to_play = games_to_play;
        games[i].max_moves = 20;
        //games[i].do_a_move = do_nonrandom_move;
        games[i].do_a_move = do_random_move;
        games[i].fen = DEFAULT_FEN;
        games[i].game_id = i + 1;

        jobs[i].data = games + i;
        jobs[i].task = play_chess;
    }

    iteration = 0;
    while (iteration < max_iterations) {
        printf("iteration %d\n", ++iteration);
        for (i = 0; i < nr_jobs; i++) {
            put_new_job(jobs + i);
        }

        /* wait for jobs to finish */
        while (get_jobs_left() > 0 || get_jobs_in_progess() > 0)
            usleep(1000 * 10); // sleep 10 ms

        if (selection_function == 0){
            for(j = 0; j < nr_selections;j++)
                natural_selection();
        }
        if(selection_function == 1) {
            best = get_best_ai(games, nr_jobs, -1);
            printf("best: %d\n", best);
            for (i = 0; i < nr_jobs; i++) {
              if( games[best].games_to_play > games[best].ai->nr_games_played)
                break;          
              if (i == best)
                    continue;
                if (get_score(games[i].ai) < get_score(games[best].ai)){
                    printf("mutating ai%d (score %f, %d wins, %d games) from ai%d (score %f, %d wins)\n",
                            i, get_score(games[i].ai), games[i].ai->nr_wins,games[i].ai->nr_games_played,  best, get_score(games[best].ai), games[best].ai->nr_wins);
                    mutate(games[i].ai, games[best].ai);
                }
                else
                     printf("not ting ai%d (score %f, %d wins, %d games) from ai%d (score %f, %d wins)\n",
                            i, get_score(games[i].ai), games[i].ai->nr_wins,games[i].ai->nr_games_played,  best, get_score(games[best].ai), games[best].ai->nr_wins);

            }
      
            //clear_score(games[best].ai);
           // games[best].ai->nr_wins/=2;
            //games[best].ai->nr_losses/=2;
           // games[best].ai->nr_games_played/=2;
           // games[best].ai->positive_reward/=2;

        }
    }
       dump_ai("ai.aidump", games[best].ai);
        clear_score(games[best].ai);

    return 0;
}
