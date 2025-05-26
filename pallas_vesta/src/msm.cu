#ifndef __MSM_CUH
#define __MSM_CUH

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#include "./../include/Point.cuh"

#define debug 1

#define WINDOW_SIZE 16
#define NUM_BITS 256
#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

struct bucket
{
    size_t count = 0;
    Point **points;
    __host__ void init(size_t num_point)
    {
        CUDA_CHECK(cudaMalloc(&points, sizeof(Point *) * num_point));
    }
    __device__ void insert(Point *address)
    {
        points[atomicAdd((unsigned long long *)&count, 1)] = address;
    }
};

__global__ void process_scalars(Scalar *sc, Point *points, size_t num_points, bucket *bucket, int current_window)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < num_points)
    {
        size_t bindex = 0;
        size_t start = current_window * WINDOW_SIZE; 
        size_t end = start + WINDOW_SIZE;
        for(size_t i = start, j = 0; i < end; i++, j++)
        {
            if(sc[idx].test_bit(i))
            {
                bindex |= (1<<j);
            }
        }
        // size_t b_num = sc[idx].get_bits_as_uint32((current_window + 1) * WINDOW_SIZE, current_window * WINDOW_SIZE);
        bucket[bindex].insert(&points[idx]);
        idx += stride;
    }
}

void construct_buckets(size_t num_points, bucket *buckets)
{
    for (size_t i = 0; i < (size_t)1 << WINDOW_SIZE; i++)
        buckets[i].init(num_points);
}

__global__ void sum_buckets()
{
}

__global__ void accumulate_result()
{
}
#if debug
__global__ void print_bucket(bucket *buckets)
{
    for (size_t i = 0; i < ((size_t)1 << WINDOW_SIZE); i++)
        if (buckets[i].count)
            printf("Bucket: %lu, Count: %lu\n", i, buckets[i].count);
}

#endif
void cuda_pippenger_msm(Point *points, Scalar *scalars, size_t num_points)
{
    int num_windows = (NUM_BITS + WINDOW_SIZE - 1) / WINDOW_SIZE;
    bucket *buckets, *d_bucket;
    buckets = new bucket[(size_t)1 << WINDOW_SIZE];
    CUDA_CHECK(cudaMalloc(&d_bucket, sizeof(bucket) * ((size_t)1 << WINDOW_SIZE)));
    CUDA_CHECK(cudaDeviceSynchronize());
    construct_buckets(num_points, buckets);

    CUDA_CHECK(cudaMemcpy(d_bucket, buckets, sizeof(bucket) * ((size_t)1 << WINDOW_SIZE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < num_windows ; i++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        process_scalars<<<512, 32>>>(scalars, points, num_points, d_bucket, i);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Time taken: %f\n", ms);

#if debug
        print_bucket<<<1, 1>>>(d_bucket);
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
        // reset bucket
        CUDA_CHECK(cudaMemcpy(d_bucket, buckets, sizeof(bucket) * ((size_t)1 << WINDOW_SIZE), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

#endif