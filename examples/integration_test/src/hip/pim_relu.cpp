/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#include "test_relu.h"

TEST_F(PimReluTestFixture, hip_pim_relu_128x1024)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(128 * 1024) == 0);
}
TEST_F(PimReluTestFixture, hip_pim_relu_1x1024)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(1 * 1024) == 0);
}
TEST_F(PimReluTestFixture, hip_pim_relu_512x1024)
{
    SetUp(RT_TYPE_HIP);
    EXPECT_TRUE(ExecuteTest(512 * 1024) == 0);
}
