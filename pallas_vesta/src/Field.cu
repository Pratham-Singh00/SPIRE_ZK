#ifndef __FIELD_CU
#define __FIELD_CU

#include "./../include/Field.cuh"

#include "./../utils/field-helper.cuh"
#include "./../constants/pasta.cuh"

// Constructor without any argument
__device__ Field::Field()
{
    this->data[0] = 0;
    this->data[1] = 0;
    this->data[2] = 0;
    this->data[3] = 0;
}

__device__ Field::Field(const u_int64_t *uint64_le, size_t len)
{
    copy_limbs(data, uint64_le, len);
    encode_montgomery();
}

__device__ Field::Field(uint64_t val) 
{
    this->data[0] = val;
    this->data[1] = 0;
    this->data[2] = 0;
    this->data[3] = 0;
    encode_montgomery();
}

__device__ Field::~Field()
{
#ifdef __CUDA_ARCH__
    free(data);
#else
    cudaFree(data);
#endif
}

// Relational Operators
__device__ bool Field::operator==(const Field &other)
{
    return equal(data, other.data, LIMBS);
}
__device__ bool Field::operator==(const Field &other) const
{
    return equal(data, other.data, LIMBS);
}
__device__ bool Field::operator!=(const Field &other)
{
    return !(operator==(other));
}
__device__ bool Field::operator!=(const Field &other) const
{
    return !(operator==(other));
}

__device__ bool is_greater_than_or_equal(const u_int64_t *a, const u_int64_t *b)
{
    for (int i = LIMBS - 1; i >= 0; i--)
    {
        if (a[i] > b[i])
            return true;
        if (a[i] < b[i])
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
    for (int i = LIMBS - 1; i >= 0; i--)
    {
        if (a[i] < b[i])
            return true;
        if (a[i] > b[i])
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

// Arithmatic operators
__device__ Field &Field::operator+=(const Field &other)
{
    add_limbs(this->data, this->data, other.data, LIMBS);
    conditional_subtract(this->data, pallas::MODULUS, LIMBS);
    return *this;
}
__device__ Field &Field::operator-=(const Field &other)
{
    __uint64_t res[LIMBS];
    bool borrow = sub_limbs(res, this->data, other.data, LIMBS);
    if (borrow)
    {
        add_limbs(this->data, pallas::MODULUS, this->data, LIMBS);
        sub_limbs(this->data, this->data, other.data, LIMBS);
    }
    else
    {
        copy_limbs(this->data, res, LIMBS);
    }
    conditional_subtract(this->data, pallas::MODULUS, LIMBS);
    return *this;
}
__device__ Field &Field::operator*=(const Field &other)
{
    __uint64_t res[2 * LIMBS];
    mont_mul(res, this->data, other.data, pallas::MODULUS, pallas::INV, LIMBS);
    conditional_subtract(res, pallas::MODULUS, LIMBS);
    copy_limbs(data, res, LIMBS);
    return *this;
}


__device__ Field Field::operator+(const Field &other)
{
    Field result;
    add_limbs(result.data, this->data, other.data, LIMBS);
    conditional_subtract(result.data, pallas::MODULUS, LIMBS);
    return result;
}
__device__ Field Field::operator+(const Field &other) const
{
    Field result;
    add_limbs(result.data, this->data, other.data, LIMBS);
    conditional_subtract(result.data, pallas::MODULUS, LIMBS);
    return result;
}
__device__ Field Field::operator-(const Field &other)
{
    Field result(*this);
    result -= other;
    return result;
}
__device__ Field Field::operator-(const Field &other) const
{
    Field result(*this);
    result -= other;
    return result;
}
__device__ Field Field::operator*(const Field &other)
{
    Field result(*this);
    result *= other;
    return result;
}
__device__ Field Field::operator*(const Field &other) const
{
    Field result(*this);
    result *= other;
    return result;
}
__device__ Field Field::operator*(const Scalar &other) 
{
    Field o(other.data);
    Field result(*this);
    result *= o;
    return result;
}
// Negation operator
__device__ Field Field::operator-()
{
    if (this->is_zero())
        return *this;

    Field result;
    sub_limbs(result.data, pallas::MODULUS, this->data, LIMBS);
    return result;
}
__device__ Field Field::operator-() const
{
    if (this->is_zero())
        return *this;

    Field result;
    sub_limbs(result.data, pallas::MODULUS, this->data, LIMBS);
    return result;
}

// double the field element
__device__ Field Field::dbl()
{
    Field result;
    for (size_t i = LIMBS - 1; i >= 1; i--)
        result.data[i] = (this->data[i] << 1) | (this->data[i - 1] >> (64 - 1));

    result.data[0] = this->data[0] << 1;
    conditional_subtract(result.data, pallas::MODULUS, LIMBS);
    return result;
}
// square of the field element
__device__ Field Field::squared()
{
    return (*this) * (*this);
}
__device__ Field Field::squared() const
{
    return (*this) * (*this);
}
// clear the set Field values
__device__ void Field::clear()
{
    for (int i = 0; i < LIMBS; i++)
        data[i] = 0;
}
// Check if equal to zero
__device__ bool Field::is_zero()
{
    return is_zero_limbs(data, LIMBS);
}
__device__ bool Field::is_zero() const
{
    return is_zero_limbs(data, LIMBS);
}

// go to montgomery representation
__device__ inline void Field::encode_montgomery()
{
    mont_encode(this->data, this->data, pallas::R2, pallas::MODULUS, pallas::INV, LIMBS);
}
// get out of montgomery representation
__device__ inline void Field::decode_montgomery()
{
    mont_decode(this->data, this->data, pallas::MODULUS, pallas::INV, LIMBS);
}
// find the inverse of the field element
__device__ Field Field::inverse()
{
    Field base(*this);

    Field result = one(); // Initialize result to 1

    // Exponent: p - 2
    // BLS12-381 modulus - 2
    __uint64_t exponent[LIMBS];
    copy_limbs(exponent, pallas::MODULUS, LIMBS);
    exponent[0] -= 2;

    // Perform exponentiation using square-and-multiply
    #pragma unroll 1
    for (int i = 5; i >= 0; --i) {
        #pragma unroll 1
        for (int bit = 63; bit >= 0; --bit) {
            result = result.squared();
            if ((exponent[i] >> bit) & 1) {
                result = result * base;
            }
        }
    }

    return result;
}

__device__ Field Field::one()
{
    Field result;
    copy_limbs(result.data, pallas::R, LIMBS);
    return result;
}


__device__ Field Field::zero()
{
    return Field();
}

__host__ __device__ void Field::print()
{
    for (int i = LIMBS - 1; i >= 0; i--)
        printf("%016lx ", data[i]);
    printf("\n");
}


__device__ Field Field::as_scalar() 
{
    this->decode_montgomery();
    return *this;
}
#endif