#ifndef __MAIN_RUN
#define __MAIN_RUN

#include <stdio.h>
#include "./../include/Point.cuh"
#include "./../include/Field.cuh"
#include "./../include/Scalar.cuh"

__global__ void check_objects()
{
    Scalar a(10);
    a.print();
    Field b(100);
    b.print();
    Point x;
    x = x.one();
    x.print();
}

int main()
{
    printf("Hello world\n");
    check_objects<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

#endif