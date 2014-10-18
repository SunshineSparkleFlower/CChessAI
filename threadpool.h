#ifndef __THREADPOOL_H
#define __THREADPOOL_H

struct job {
    void (*task)(void *arg);
    void *data;
};

extern void put_new_job(struct job *j);
extern void init_threadpool(int pool_size);
extern void set_free_function(void (*f)(void *));
extern void shutdown_threadpool(void);

#endif
