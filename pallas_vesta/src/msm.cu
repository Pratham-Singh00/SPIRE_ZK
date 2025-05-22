#ifndef __MSM_CUH
#define __MSM_CUH

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#include "./../include/Point.cuh"

#define C 13
#define NUM_BUCKETS (((size_t)1<<C) -1)
#define SCALAR_BITS 256
#define SEGMENTS ((SCALAR_BITS + C -1) / C)

__global__ void msm_kernel(
    const Scalar* scalars,
    const Point* bases,
    Point* buckets,
    int n,
    int segment
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t digit = scalars[idx].get_bits_as_uint32((segment+1)*C - 1, segment*C);// get_window(&scalars[idx], segment);
    if (digit == 0) return;

    int bucket_id = digit - 1;
    buckets[bucket_id] = buckets[bucket_id] + bases[idx];
}


__global__ void process_scalar_kernel(const Scalar *scalars, int n)
{

}

void cuda_msm(Scalar *scalars, Point *points, int n, Point *result)
{
    Scalar* d_scalars;
    Point* d_bases;
    Point* d_buckets;
    size_t *d_bucket_length;
    size_t *d_base_ptr;
    

    cudaMalloc(&d_scalars, sizeof(Scalar) * n);
    cudaMalloc(&d_bases, sizeof(Point) * n);
    cudaMalloc(&d_buckets, sizeof(Point) * NUM_BUCKETS);

    cudaMalloc(&d_bucket_length, sizeof(size_t)* NUM_BUCKETS);

    cudaMemcpy(d_scalars, scalars, sizeof(Scalar) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bases, points, sizeof(Point) * n, cudaMemcpyHostToDevice);
    cudaMemset(d_buckets, 0, sizeof(Point) * NUM_BUCKETS);
    cudaMemset(d_bucket_length, 0, sizeof(size_t) * NUM_BUCKETS);


    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Point acc;

    // for (int seg = SEGMENTS - 1; seg >= 0; --seg) {
    //     for (int i = 0; i < C; i++) {
    //         acc = acc.dbl();
    //     }

    //     cudaMemset(d_buckets, 0, sizeof(Point) * NUM_BUCKETS);
    //     msm_kernel<<<blocks, threads>>>(d_scalars, d_bases, d_buckets, n, seg);
    //     cudaDeviceSynchronize();

    //     Point buckets[NUM_BUCKETS];
    //     cudaMemcpy(buckets, d_buckets, sizeof(Point) * NUM_BUCKETS, cudaMemcpyDeviceToHost);

    //     Point running_sum;
    //     for (int i = NUM_BUCKETS - 1; i >= 0; --i) {
    //         running_sum = running_sum + buckets[i];
    //         acc = acc + running_sum;
    //     }
    // }

    // *result = acc;

    cudaFree(d_scalars);
    cudaFree(d_bases);
    cudaFree(d_buckets);
}


#endif