#ifndef __MAIN_RUN
#define __MAIN_RUN

#include <stdio.h>
#include <iostream>

#include "./../include/Point.cuh"

#include "./msm.cu"

#include "./../constants/pasta_17.cuh"

#define debug 1

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


__global__ void init_points_from_sage(Point *p, size_t num)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < num)
    {
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
__global__ void print_point(uint64_t *x, uint64_t *y, uint64_t *z)
{
    Field fx(x);
    Field fy(y);
    Field fz(z);
    Point p(fx, fy, fz);
    p.to_affine();
    p.print();
}


int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        printf("Please enter the MSM scales (e.g. 20 represents 2^20) \n");
        return 1;
    }

    int log_size = atoi(argv[1]);

    size_t num_v = (size_t)(1 << log_size);

    Point *points;
    Scalar *scalars;
    CUDA_CHECK(cudaMalloc(&points, sizeof(Point) * num_v));
    CUDA_CHECK(cudaMalloc(&scalars, sizeof(Scalar) * num_v));

    // CUDA_CHECK(cudaMemcpy(points, sage_points, sizeof(uint64_t) * num_v * 4 * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(
        points,              // dst
        sizeof(Point), // dst pitch (128 bytes)
        sage_points,                // src
        96,                   // src pitch (96 bytes)
        96,                   // width (bytes to copy per item)
        num_v,                    // height (items)
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(scalars, sage_scalars, sizeof(uint64_t) * num_v * 4, cudaMemcpyHostToDevice));

    init_points_from_sage<<<512, 128>>>(points, num_v);
    CUDA_CHECK(cudaDeviceSynchronize());


    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaEventSynchronize(start));


    cuda_pippenger_msm(points, scalars, num_v);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Elapsed time: %f ms\n", elapsedTime);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("Sage Result:\n");
    uint64_t *x, *y, *z;
    CUDA_CHECK(cudaMalloc(&x, sizeof(uint64_t) * 4));
    CUDA_CHECK(cudaMalloc(&y, sizeof(uint64_t) * 4));
    CUDA_CHECK(cudaMalloc(&z, sizeof(uint64_t) * 4));
    CUDA_CHECK(cudaMemcpy(x, sage_msm_result[0], sizeof(uint64_t) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y, sage_msm_result[1], sizeof(uint64_t) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(z, sage_msm_result[2], sizeof(uint64_t) * 4, cudaMemcpyHostToDevice));


    print_point<<<1, 1>>>(x, y, z);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t t = cudaGetLastError();
    if (t != cudaSuccess)
    {
        printf("Cuda Error: %s \n", cudaGetErrorString(t));
        printf("Peek: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    return 0;
}

#endif