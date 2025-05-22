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
    Point g;
    g = g.one();
    g.print();
}

int main(int argc, char* argv[])
{
    printf("Hello world\n");
    if (argc < 2) {
		printf("Please enter the MSM scales (e.g. 20 represents 2^20) \n");
		return 1;
	}

    // int log_size = atoi(argv[1]);
    // size_t num_v = (size_t) (1 << log_size);

    // int deviceCount;
    // cudaGetDeviceCount( &deviceCount );
    // CUTThread  thread[deviceCount];

    // Mem device[deviceCount];

    // for(int i=0; i< deviceCount; i++)
    //     start_thread(init_msm_params, &device[i]);

    test_one<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

#endif