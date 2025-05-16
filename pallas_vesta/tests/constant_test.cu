#ifndef __CONSTANT_UNIT_TEST
#define __CONSTANT_UNIT_TEST

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "./../constants/pasta.cuh"
#include "./../utils/field-helper.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>


// Parse a big decimal string to std::vector<uint8_t> in little-endian base-256 form
std::vector<uint8_t> parse_decimal_to_bytes(const std::string& dec_str) {
    std::vector<uint8_t> result;

    std::string num = dec_str;
    while (!num.empty() && num != "0") {
        std::string quotient;
        int remainder = 0;
        for (char c : num) {
            int digit = c - '0';
            int value = remainder * 10 + digit;
            quotient += ('0' + value / 256);
            remainder = value % 256;
        }
        result.push_back(static_cast<uint8_t>(remainder));
        // Remove leading zeros
        size_t first_nonzero = quotient.find_first_not_of('0');
        num = (first_nonzero == std::string::npos) ? "" : quotient.substr(first_nonzero);
    }
    return result;
}

// Convert byte array to 4 uint64_t limbs (little endian)
std::vector<uint64_t> bytes_to_limbs_le(const std::vector<uint8_t>& bytes) {
    std::vector<uint64_t> limbs(4, 0);
    for (size_t i = 0; i < bytes.size() && i < 32; ++i) {
        size_t limb_index = i / 8;
        limbs[limb_index] |= static_cast<uint64_t>(bytes[i]) << (8 * (i % 8));
    }
    return limbs;
}

std::string readFile(const char* filename) {
    std::ifstream file(filename);
    std::string decimal_str;
    if (!file || !(file >> decimal_str)) {
        std::cerr << "Failed to read number from file\n";
        return "0";
    }
    return decimal_str;
}

uint64_t* get_data_from_file(const char* filename) {
    std::string content = readFile(filename);
    std::vector<uint8_t> bytes = parse_decimal_to_bytes(content);
    std::vector<uint64_t> limbs = bytes_to_limbs_le(bytes);
    return limbs.data();
}

class CurveConstants: public ::testing::Test
{
public:
    CurveConstants()
    {

    }
    ~CurveConstants() override
    {

    }
    void SetUp() override
    {

    }
    void TearDown() override
    {

    }
};

TEST_F(CurveConstants, check_R_Pallas) 
{
    int ret = std::system("sage -python constant_sage.py r pallas");
    // if (ret != 0) 
    // {
    //     ASSERT_TRUE(false);
    // }
    uint64_t *out = get_data_from_file("sage_constant_output.txt");

    for(int i=0;i<LIMBS; i++)
    {
        printf("%016x ", out[i]);
    }
    printf("\n");
    for(int i=0;i<LIMBS; i++)
    {
        printf("%016x ", pallas::R[i]);
    }
    ASSERT_TRUE(equal(out, pallas::R, LIMBS));
}
#endif