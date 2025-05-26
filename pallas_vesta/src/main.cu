#ifndef __MAIN_RUN
#define __MAIN_RUN

#include <stdio.h>
#include <iostream>

#include "./../include/Field.cuh"
#include "./../include/Point.cuh"

#include "./msm.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

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

    Point g, s, t;
    g = g.one();
    printf("One\n");
    g.print();
    printf("Well formed: %d\n", g.is_well_formed());

    s = g.dbl();
    printf("Two\n");
    s.to_affine();
    s.print();
    printf("Well formed: %d\n", s.is_well_formed());
    s = s + g;
    printf("Three\n");
    s.to_affine();
    s.print();

}
__global__ void init_points_scalars(Scalar *scalar, Point *point, size_t num)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while(idx < num)
    {
        scalar[idx] = scalar[idx].random();
        point[idx] = point[idx].random() * idx;
        idx += stride;
    }
}
__global__ void Scalar_test()
{
    Scalar sc;
    sc = sc.random();
    sc.print();
    for(int i = 0; i<16; i++)
    {
        printf("%d: %08x\n", i, sc.get_bits_as_uint32((i+1)*16, i*16));
    }
}

int main(int argc, char* argv[])
{

    if (argc < 2) {
		printf("Please enter the MSM scales (e.g. 20 represents 2^20) \n");
		return 1;
	}

    int log_size = atoi(argv[1]);

    size_t num_v = (size_t) (1 << log_size);

    Point *points;
    Scalar *scalars;
    CUDA_CHECK(cudaMalloc(&points, sizeof(Point)*num_v));
    CUDA_CHECK(cudaMalloc(&scalars, sizeof(Scalar)*num_v));

    init_points_scalars<<<512, 128>>>(scalars, points, num_v);
    CUDA_CHECK(cudaDeviceSynchronize());

    Scalar_test<<<1,1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_pippenger_msm(points, scalars, num_v);


    cudaError_t t = cudaGetLastError();
    if(t != cudaSuccess)
    {
        printf("Cuda Error: %s \n", cudaGetErrorString(t));
        printf("Peek: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    return 0;
}

#endif