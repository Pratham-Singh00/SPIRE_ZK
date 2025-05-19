#ifndef __FIELD_UNIT_TEST
#define __FIELD_UNIT_TEST

#include <gtest/gtest.h>

class FieldTests : public ::testing::Test
{
public:
    FieldTests()
    {
    }
    ~FieldTests() override
    {
    }
    void SetUp() override
    {
    }
    void TearDown() override
    {
    }
};

TEST_F(FieldTests, check_equal_operator)
{
    ASSERT_TRUE(true);
}

#endif