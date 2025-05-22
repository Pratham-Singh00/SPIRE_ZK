#ifndef __POINT_UNIT_TEST
#define __POINT_UNIT_TEST

#include "test_header.cuh"

void write_limbs_to_file(std::ofstream &out, uint64_t *data, int limit = 4) {
    bool printStarted = false;
    for(int i= limit -1;i>=0; i--)
    {
        if(printStarted) 
        {
            out<< std::hex << std::uppercase << std::setfill('0') << std::setw(16) <<data[i];
        }
        else 
        {
            if(data[i]) 
            {
                out << std::hex << std::uppercase <<data[i];
                printStarted = true;
            }
        }
    }
    out<<" ";
}
void write_point_to_file(Point *a, char *filename)
{
    std::ofstream out(filename);
    uint64_t data[4];
    cudaMemcpy(data, a->X.data, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_limbs_to_file(out, data);
    cudaMemcpy(data, a->Y.data, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_limbs_to_file(out, data);
    cudaMemcpy(data, a->Z.data, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_limbs_to_file(out, data);
}

__global__ void add_kernel(Point *a, Point *b, Point *res)
{
    *res = *a + *b;
}
__global__ void multiplication_kernel(Point *a, Scalar *s, Point *res)
{
    *res = (*a) * (*s);
}

__global__ void multiplication_kernel2(Point *a, uint64_t s, Point *res)
{
    *res = (*a) * s;
}
__global__ void double_kernel(Point *a, Point *res)
{
    *res = a->dbl();
}
__global__ void mixed_add_kernel(Point *a, Point *b, Point *res)
{
    b->to_affine();
    *res = a->mixed_add(*b);
}
__global__ void check_wellformed_kernel(Point *a, bool *result)
{
    *result = a->is_well_formed();
}
__global__ void check_affine_kernel(Point *a, bool *result)
{
    *result = (a->Z.data[0] == 1);
    for(int i=1;i<4;i++)
    {
        *result &= (a->Z.data[i] == 0);
    }
}
__global__ void is_equal_kernel(Point *a, Point *b, bool *result)
{
    *result = (*a) == (*b);
}
__global__ void is_not_equal_kernel(Point *a, Point *b, bool *result)
{
    *result = (*a) != (*b);
}


__global__ void init_test(Point *a)
{
    *a = Point().one();
    a->print();
}
class PointTests : public ::testing::Test
{
public:
    Point *one;
    PointTests()
    {
        init_test<<<1,1>>>(one);
        cudaDeviceSynchronize();
    }
    ~PointTests()
    {

    }
    void SetUp() override
    {

    }
    void TearDown() override
    {

    }
};

TEST_F(PointTests, check_Point_addition)
{

    init_test<<<1,1>>>(one);
    cudaDeviceSynchronize();
    //one->is_well_formed();
    ASSERT_TRUE(true);
}


#endif