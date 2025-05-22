#ifndef __MAIN_RUN
#define __MAIN_RUN

#include <stdio.h>
#include "./../include/Point.cuh"



#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define  CUT_THREADEND

//Create thread
CUTThread start_thread(CUT_THREADROUTINE func, void * data){
    pthread_t thread;
    pthread_create(&thread, NULL, func, data);
    return thread;
}

//Wait for thread to finish
void end_thread(CUTThread thread){
    pthread_join(thread, NULL);
}

//Destroy thread
void destroy_thread( CUTThread thread ){
    pthread_cancel(thread);
}

//Wait for multiple threads
void wait_for_threads(const CUTThread * threads, int num){
    for(int i = 0; i < num; i++)
        end_thread( threads[i] );
}

void* init_msm_params(void* params)
{

}

struct Mem
{
    size_t device_id;
    void* mem;
};

__global__ void test_one()
{

    printf("Debug 0\n");
    Point g;
    g = g.one();
    printf("debug 0\n");
    g.print();
    printf("Debug 1\n");
    Point s;
    printf("Debug 2\n");
    s = g.dbl();
    s.print();
    s = s + g;
    s.print();
}

int main(int argc, char* argv[])
{
    printf("Hello world\n");
   
    test_one<<<1,1>>>();
    cudaDeviceSynchronize();
    cudaError_t t = cudaGetLastError();
    if(t != cudaSuccess)
    {
        printf("Cuda Error: %s \n", cudaGetErrorString(t));
        printf("Peek: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    return 0;
}

#endif