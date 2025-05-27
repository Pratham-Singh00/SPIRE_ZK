#ifndef __MAIN_RUN
#define __MAIN_RUN

#include <stdio.h>
#include <iostream>

#include "./../include/Field.cuh"
#include "./../include/Point.cuh"

#include "./msm.cu"

#include "./../constants/msm_sage_values_2.cuh"

#define debug 1

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
__global__ void init_points_scalars(Scalar *scalar, Point *point, size_t num, uint64_t *sage_scalars, uint64_t *sage_points)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while(idx < num)
    {
        scalar[idx] = Scalar(&sage_scalars[idx*4], 4);
        point[idx].X = Field(&sage_points[2*4*idx]);
        point[idx].Y = Field(&sage_points[2*4*idx + 4]);
        point[idx].Z = point[idx].Z.zero();
        point[idx].to_affine();
        idx += stride;
    }
}

#if debug

__global__ void check_construction(Point *point, Scalar *scalar)
{
    for(int i=0; i< 10; i++)
    {
        point[i].print();
        scalar[i].print();
    }
}

#endif

__global__ void init_points_from_sage(Point *p, size_t num)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while(idx < num) {
        p[idx].X.encode_montgomery();
        p[idx].Y.encode_montgomery();
        p[idx].Z.encode_montgomery();

        idx += stride;
    }
}

__global__ void init_sage_result(Point *p, const uint64_t *x, const uint64_t *y)
{
    p->X = Field(x);
    p->Y = Field(y);
    p->Z = p->Z.one();
    p->print();
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

    CUDA_CHECK(cudaMemcpy(points, sage_points, sizeof(uint64_t)*num_v*4*3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scalars, sage_scalars, sizeof(uint64_t)*num_v*4, cudaMemcpyHostToDevice));

    init_points_from_sage<<<512,128>>>(points, num_v);
    CUDA_CHECK(cudaDeviceSynchronize());

    // check_construction<<<1,1>>>(points, scalars);

    cuda_pippenger_msm(points, scalars, num_v);
    CUDA_CHECK(cudaDeviceSynchronize());

    Point *sage_res;
    CUDA_CHECK(cudaMalloc(&sage_res, sizeof(Point)));
    // init_sage_result<<<1,1>>>(sage_res, sage_msm_result[0], sage_msm_result[1]);
    // CUDA_CHECK(cudaDeviceSynchronize());
    printf("Sage Result:\n");
    for(int i=0; i< 2; i++)
    {
        if(!i) printf("X = \n");
        else printf("Y = \n");
        for(int j=3; j>=0 ; j--)
            printf("%016lx ", sage_msm_result[i][j]);
        printf("\n");
    }
    cudaError_t t = cudaGetLastError();
    if(t != cudaSuccess)
    {
        printf("Cuda Error: %s \n", cudaGetErrorString(t));
        printf("Peek: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    return 0;
}

#endif