#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include "common.h"
#include "board.h"
#include "AI.h"
#include "threadpool.h"

struct stats {
    int ai_wins;
};

struct game_struct {
    AI_instance_t *ai;
    int games_to_play, max_moves;
    int (*do_a_move)(board_t *);
    char *fen;

    int game_id;
    struct stats *stats;
};

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
    struct stats *stats = game->stats;
    int games_to_play, nr_games, max_moves, moves, ret, ai_wins;
    int (*do_a_move)(board_t *);
    AI_instance_t *ai;
    board_t *board;

    ai = game->ai;
    games_to_play = game->games_to_play;
    max_moves = game->max_moves;
    do_a_move = game->do_a_move;
    ai_wins = 0;

    printf("starting game %d\n", game->game_id);

    for (nr_games = 0; nr_games < games_to_play; nr_games++) {
        board = new_board(game->fen);
        for (moves = 0; moves < max_moves; moves++) {
            ret = do_best_move(ai, board);
            if(ret == 0)
                break;
            else if (ret == -1)
                break;

            ret = do_a_move(board);
            if(ret == 0)
                break;
            else if(ret == -1) {
                ai_wins++;
                break;
            }
        }
        free_board(board);
    }

    stats->ai_wins = ai_wins;
}

int get_best_ai(struct stats *s, int n)
{
    int i, best = 0;;

    for (i = 1; i < n; i++) {
        if (s[i].ai_wins > s[best].ai_wins)
            best = i;
    }

    return best;
}

int main(int argc, char *argv[])
{
    int nr_threads, nr_jobs, i, best;
    struct game_struct *games;
    struct stats *stats;
    struct job *jobs;

    nr_threads = argc > 1 ? atoi(argv[1]) : 2;
    nr_jobs = argc > 2 ? atoi(argv[2]) : 2;

    if (nr_threads == 0 || nr_jobs == 0) {
        printf("threads or jobs cannot be 0\n");
        printf("USAGE: %s <nr threads> <nr jobs>\n", argv[0]);
        return 1;
    }

    init_threadpool(nr_threads);
    init_magicmoves();

    jobs = malloc(nr_jobs * sizeof(struct job));
    games = malloc(nr_jobs * sizeof(struct game_struct));
    stats = malloc(nr_jobs * sizeof(struct stats));

    for (i = 0; i < nr_jobs; i++) {
        games[i].ai = ai_new();
        games[i].games_to_play = 1000;
        games[i].max_moves = 100;
        games[i].do_a_move = do_nonrandom_move;
        games[i].fen = DEFAULT_FEN;
        games[i].game_id = i + 1;
        games[i].stats = stats + i;

        jobs[i].data = games + i;
        jobs[i].task = play_chess;
    }

    while (1) {
        for (i = 0; i < nr_jobs; i++) {
            stats[i].ai_wins = 0;
            put_new_job(jobs + i);
        }

        /* wait for jobs to finish */
        while (get_jobs_left() > 0 || get_jobs_in_progess() > 0)
            usleep(1000 * 10); // sleep 10 ms

        best = get_best_ai(stats, nr_jobs);
        for (i = 0; i < nr_jobs; i++) {
            if (i == best)
                continue;

            printf("mutating ai%d (%d wins) from ai%d (%d wins)\n",
                    i, stats[i].ai_wins, best, stats[best].ai_wins);
            mutate(games[i].ai, games[best].ai);
        }
    }

    return 0;
}
