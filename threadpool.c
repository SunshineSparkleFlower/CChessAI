#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <pthread.h>
#include "threadpool.h"

struct job_list {
    struct job_list *next, *prev;
    struct job *job;
};

static volatile int run = 1; /* is set to 0 if threads are to terminate */
static volatile int num_jobs = 0;
static volatile struct job_list jobs;

/* used to free job->data */
static void (*free_function)(void *) = NULL;
static pthread_mutex_t jobs_lock;
static pthread_t *threads;
static int num_threads = 0;

/* must be called in a locked context */
static inline void add_tail(struct job_list *j)
{
    struct job_list *tail;

    tail = jobs.prev;
    j->next = tail->next;
    j->prev = tail;
    tail->next = j;
    jobs.prev = j;
}

/* must be called in a locked context */
static inline void unlink_node(struct job_list *j)
{
    j->next->prev = j->prev;
    j->prev->next = j->next;
}

void put_new_job(struct job *j)
{
    struct job_list *new;
    pthread_mutex_lock(&jobs_lock);

    new = malloc(sizeof(struct job_list));
    new->job = j;

    add_tail(new);
    ++num_jobs;

    pthread_mutex_unlock(&jobs_lock);
}

static struct job *get_job(void)
{
    struct job_list *tmp;
    struct job *ret;

    while (run) {
        while (num_jobs == 0 && run)
            usleep(10 * 1000); /* sleep 10 milliseconds */

        if (!run)
            break;

        pthread_mutex_lock(&jobs_lock);

        if (num_jobs != 0) {
            tmp = jobs.next;
            unlink_node(tmp);
            ret = tmp->job;

            --num_jobs;
            free(tmp);

            pthread_mutex_unlock(&jobs_lock);
            return ret;
        }

        pthread_mutex_unlock(&jobs_lock);
    }

    return NULL; /* should never happen */
}

static void *loiter(void *arg)
{
    struct job *job;
    int thread_id = (long)arg;

    while (run) {
        job = get_job();
        if (job == NULL)
            break;

        printf("thread %d got a job\n", thread_id);
        job->task(job->data);

        if (free_function)
            free_function(job->data);
        free(job);
    }

    return NULL;
}

void init_threadpool(int pool_size)
{
    int i;

    jobs.next = jobs.prev = &jobs;
    pthread_mutex_init(&jobs_lock, NULL);

    threads = malloc(pool_size * sizeof(pthread_t));
    for (i = 0; i < pool_size; i++)
        pthread_create(threads + i, NULL, loiter, (void *)(long)i + 1);

    num_threads = pool_size;
}

/* register a function to be used to free job->data */
void set_free_function(void (*f)(void *))
{
    free_function = f;
}

void shutdown_threadpool(void)
{
    int i;
    struct job_list *ptr, *next;

    run = 0;

    for (i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);
    pthread_mutex_destroy(&jobs_lock);

    for (ptr = jobs.next; ptr != &jobs; ptr = next) {
        next = ptr->next;
        if (free_function)
            free_function(ptr->job->data);
        free(ptr->job);
        free(ptr);
    }
    free(threads);
}

static void hello(void *a)
{
    int id = (long)a;
    //printf("hello world (%d)\n", id);
    usleep(100*1000);
}

/*
int main(int argc, char *argv[])
{
    int i;
    struct job *jobs[100];

    for (i = 0; i < 100; i++) {
        jobs[i] = malloc(sizeof(struct job));
        jobs[i]->task = hello;
        jobs[i]->data = (void *)(long)i + 1;
    }

    init_threadpool(10);

    for (i = 0; i < 100; i++)
        put_new_job(jobs[i]);

    getchar();

    shutdown_threadpool();

    return 0;
}
*/
