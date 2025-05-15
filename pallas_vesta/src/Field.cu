#ifndef __FIELD_CU
#define __FIELD_CU

#include "./../include/Field.cuh"

// Constructor without any argument
__host__ __device__ Field::Field()
{
    this->data[0] = 0;
    this->data[1] = 0;
    this->data[2] = 0;
    this->data[3] = 0;
}

__device__ void copy_data(u_int64_t *a, const u_int64_t *b)
{
    #pragma unroll
    for(int i=0;i<4;i++)
        a[i] = b[i];
}

__host__ __device__ Field::Field(u_int64_t *uint64_le, size_t len = 4)
{
    copy_data(data, uint64_le);
    encode_montgomery();
}

__host__ __device__ Field::~Field()
{
    #ifdef __CUDA_ARCH__
        free(data);
    #else 
        cudaFree(data);
    #endif
}

__device__ bool checkEqual(const u_int64_t *a, const u_int64_t *b)
{
    for(int i=0;i<4; i++)
    {
        if(a[i] != b[i])
            return false;
    }
    return true;
}
// Relational Operators
__device__ bool Field::operator==(const Field &other)
{
    return checkEqual(data, other.data);
}
__device__ bool Field::operator==(const Field &other) const
{
    return checkEqual(data, other.data);
}
__device__ bool Field::operator!=(const Field &other)
{
    return !checkEqual(data, other.data);
}
__device__ bool Field::operator!=(const Field &other) const
{
    return !checkEqual(data, other.data);
}

__device__ bool is_greater_than_or_equal(const u_int64_t *a, const u_int64_t *b)
{
    for(int i=3; i>=0; i++)
    {
        if (a[i]<b[i])
            return false;
    }
    return true;
}
__device__ bool Field::operator>=(const Field &other)
{
    return is_greater_than_or_equal(data, other.data);
}
__device__ bool Field::operator>=(const Field &other) const
{
    return is_greater_than_or_equal(data, other.data);
}

__device__ bool is_less_than_or_equal(const u_int64_t *a, const u_int64_t *b)
{
    for(int i=3; i>=0; i++)
    {
        if (a[i]>b[i])
            return false;
    }
    return true;
}

__device__ bool Field::operator<=(const Field &other)
{
    return is_less_than_or_equal(data, other.data);
}
__device__ bool Field::operator<=(const Field &other) const
{
    return is_less_than_or_equal(data, other.data);
}


// Assignment operators
__device__ Field &Field::operator=(const Field &other)
{
    copy_data(data, other.data);
    return *this;
}
__device__ Field &Field::operator=(const Field &other) const
{
    copy_data(data, other.data);
    return *this;
}

// Arithmatic operators
__device__ Field &Field::operator+=(const Field &other)
{
    
}
__device__ Field &Field::operator+=(const Field &other) const
{
    
}
__device__ Field &Field::operator-=(const Field &other)
{
    
}
__device__ Field &Field::operator-=(const Field &other) const
{
    
}
__device__ Field &Field::operator*=(const Field &other)
{
    
}
__device__ Field &Field::operator*=(const Field &other) const
{
    
}
__device__ Field Field::operator+(const Field &other)
{
    
}
__device__ Field Field::operator+(const Field &other) const
{
    
}
__device__ Field Field::operator-(const Field &other)
{
    
}
__device__ Field Field::operator-(const Field &other) const
{
    
}
__device__ Field Field::operator*(const Field &other)
{
    
}
__device__ Field Field::operator*(const Field &other) const
{
    
}

// Negation operator
__device__ Field Field::operator-()
{
    
}
__device__ Field Field::operator-() const
{
    
}

// double the field element
__device__ Field Field::dbl()
{
    
}
__device__ Field Field::dbl()
{
    
}
// square of the field element
__device__ Field Field::squared()
{
    
}
__device__ Field Field::squared() const
{
    
}
// clear the set Field values
__device__ void Field::clear()
{
    
}
// Check if equal to zero
__device__ bool Field::is_zero()
{
    
}
__device__ bool Field::is_zero() const
{
    
}

// go to montgomery representation
__device__ inline void Field::encode_montgomery()
{

}
// get out of montgomery representation
__device__ inline void Field::decode_montgomery()
{

}
// find the inverse of the field element
__device__ Field Field::inverse()
{
    
}

__host__ __device__ void Field::print()
{
    
}

#endif