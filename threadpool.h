#ifndef __THREADPOOL_H
#define __THREADPOOL_H

struct job {
    void (*task)(void *arg);
    void *data;
};

extern void put_new_job(struct job *j);
extern int get_jobs_left(void);
extern int get_jobs_in_progess(void);
extern void init_threadpool(int pool_size);
extern void set_free_function(void (*f)(void *));
extern void shutdown_threadpool(int force_kill);

#endif
