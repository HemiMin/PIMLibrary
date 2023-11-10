/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */
#include "gcn_perf.h"
#include "module.h"
#include "optim.h"
#include "variable.h"
#include "parser.h"
#include "hip_gcn.h"
#include "gcn_parser.h"

using half_float::half;
using namespace std;

PimGCNTest::PimGCNTest()
{
}

PimGCNTest::~PimGCNTest()
{
}

void PimGCNTest::prepare(float variation)
{
}

void PimGCNTest::execute_op(bool block)
{
    GCNParams params = GCNParams::get_default();
    GCNData data;
    std::string input_name("cora");
    GCNParser parser(&params, &data, input_name);
    if (!parser.parse()) {
        std::cerr << "Cannot read input: " << input_name << std::endl;
        exit(EXIT_FAILURE);
    }

    #ifdef __HIPCC__
    std::cout << "RUNNING ON GPU(HIP)" << std::endl;
    CUDAGCN cuda_gcn(params, &data);
    cuda_gcn.run();
    #else
    std::cout << "RUNNING ON CPU" << std::endl;
    GCN gcn(params, &data);
    gcn.run();
    #endif
}

void PimGCNTest::finalize() 
{ 
  //PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST); 
}

int PimGCNTest::validate(float epsilon)
{
  return 1;
}

PimGCNTestFixture::PimGCNTestFixture() {}
int PimGCNTestFixture::ExecuteTest()
{
    PimGCNTest pimGCNTest = PimGCNTest();
    pimGCNTest.prepare();

    // warmup
    pimGCNTest.execute_op(true);

    //avg_kernel_time_ = std::chrono::duration<double>::zero();
    //    Tick();
    //    pimGCNTest.execute_op(block_);
    //    Tock();
    //    avg_kernel_time_ += calculate_elapsed_time();
    //pimGCNTest.finalize();
    //calculate_avg_time();
    //calculate_gflops(pimGCNTest.get_flt_ops());
    return pimGCNTest.validate();
}

