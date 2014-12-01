#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <pthread.h>
#include "threadpool.h"
#include "common.h"

struct job_list {
    struct job_list *next, *prev;
    struct job *job;
};

struct thread_struct {
    pthread_t thread;
    thread_init_func_t init_function;
    void *init_function_arg;
    int thread_id;
};

static volatile int run = 1; /* is set to 0 if threads are to terminate */
static volatile int num_free_jobs = 0;
static volatile int jobs_in_progress = 0;
static struct job_list jobs;

/* used to free job->data */
static void (*free_function)(void *) = NULL;
static pthread_mutex_t jobs_lock;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static struct thread_struct *threads;
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
    ++num_free_jobs;

    pthread_mutex_unlock(&jobs_lock);
    pthread_cond_signal(&cond);
}

int get_jobs_left(void)
{
    return num_free_jobs;
}

int get_jobs_in_progess(void)
{
    return jobs_in_progress;
}

int get_thread_id(void)
{
    int i;

    for (i = 0; i < num_threads; i++) {
        if (threads[i].thread == pthread_self())
            return threads[i].thread_id;
    }
    return -1;
}

static struct job *get_job(void)
{
    struct job_list *tmp;
    struct job *ret = NULL;

    pthread_mutex_lock(&jobs_lock);
    while (num_free_jobs == 0 && run)
        pthread_cond_wait(&cond, &jobs_lock);

    if (run && num_free_jobs > 0) {
        tmp = jobs.next;
        unlink_node(tmp);
        ret = tmp->job;
        free(tmp);

        --num_free_jobs;
    }

    pthread_mutex_unlock(&jobs_lock);
    return ret;
}

static inline void report_start(void)
{
    pthread_mutex_lock(&jobs_lock);
    ++jobs_in_progress;
    pthread_mutex_unlock(&jobs_lock);
}

static inline void report_done(void)
{
    pthread_mutex_lock(&jobs_lock);
    --jobs_in_progress;
    pthread_mutex_unlock(&jobs_lock);
}

static void *loiter(void *arg)
{
    long i = 0;
    struct job *job;
    struct thread_struct *thread_info = arg;

    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL); 

    if (thread_info->init_function)
        thread_info->init_function(thread_info->thread_id,
                thread_info->init_function_arg);

    while (run) {
        job = get_job();
        if (job == NULL)
            break;
        ++i;

        report_start();
        job->task(job->data);
        report_done();

        if (free_function)
            free_function(job);
    }

    return (void *)i;
}

int init_threadpool(int pool_size, thread_init_func_t init_function, 
        void *arg)
{
    int i;

    jobs.next = jobs.prev = &jobs;
    pthread_mutex_init(&jobs_lock, NULL);

    threads = malloc(pool_size * sizeof(struct thread_struct));
    if (threads == NULL)
        return NULL;

    for (i = 0; i < pool_size; i++) {
        threads[i].init_function = init_function;
        threads[i].init_function_arg = arg;
        threads[i].thread_id = i;
        pthread_create(&threads[i].thread, NULL, loiter, (void *)&threads[i]);
    }

    num_threads = pool_size;
}

/* register a function to be used to free job when completed */
void set_free_function(void (*f)(void *))
{
    free_function = f;
}

void shutdown_threadpool(int force_kill)
{
    int i, count;
    struct job_list *ptr, *next;

    run = 0;
    pthread_cond_broadcast(&cond);

    for (i = 0; i < num_threads; i++) {
        if (force_kill)
            pthread_cancel(threads[i].thread);
        pthread_join(threads[i].thread, (void *)&count);
    }
    pthread_mutex_destroy(&jobs_lock);

    for (ptr = jobs.next; ptr != &jobs; ptr = next) {
        next = ptr->next;
        if (free_function)
            free_function(ptr->job);
        free(ptr);
    }
    free(threads);
}

/*
   static void hello(void *a)
   {
   int id = (long)a;
   printf("hello world (%d)\n", id);
   }

   int main(int argc, char *argv[])
   {
   int i;
   struct job *jobs[100];

   for (i = 0; i < 100; i++) {
   jobs[i] = malloc(sizeof(struct job));
   jobs[i]->task = hello;
   jobs[i]->data = (void *)(long)i + 1;
   }

   init_threadpool(20);

   for (i = 0; i < 100; i++)
   put_new_job(jobs[i]);

   getchar();

   shutdown_threadpool();

   return 0;
   }
   */
