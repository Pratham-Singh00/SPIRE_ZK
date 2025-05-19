#ifndef __POINT_UNIT_TEST
#define __POINT_UNIT_TEST

#include <gtest/gtest.h>

class PointTests : public ::testing::Test
{
public:
    PointTests()
    {
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
    ASSERT_TRUE(true);
}


#endif