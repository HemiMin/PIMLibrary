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
#include <chrono>
#include "module.h"
#include "optim.h"
#include "variable.h"
#include "parser.h"
#include "hip_gcn.h"
#include "gcn_parser.h"
#include "pim_runtime_api.h"
#include "utility/pim_profile.h"

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

void PimGCNTest::execute_op(PerformanceAnalyser* pa, bool block)
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
    CUDAGCN cuda_gcn(params, &data, pa);
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
    PimBo* wu_i = PimCreateBo(1,1,1,256, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* wu_w = PimCreateBo(1,1,256,4096, PIM_FP16, MEM_TYPE_DEVICE);
    PimBo* wu_o = PimCreateBo(1,1,1,4096, PIM_FP16, MEM_TYPE_DEVICE);
    (void)PimExecuteGemm(wu_o, wu_i, wu_w, nullptr, PimActFunc::NONE, I_X_W, nullptr);
    std::cout << "=======WarmUp end========" << std::endl;

    avg_kernel_time_ = std::chrono::duration<double>::zero();
    for (int i = 0; i < num_iter_; i++) {
      PIM_PROFILE_TICK_A(TotalTime);
      pimGCNTest.execute_op((PerformanceAnalyser*)this, true);
      PIM_PROFILE_TOCK_A(TotalTime);
    }
    calculate_avg_time();

    std::cout << "===========GCN Time===============" << std::endl;
    std::cout << "Time taken to initialize PIM : " << std::fixed << start_up_time_.count() * 1000000 << " us\n";
    std::cout << "Time taken to allocH PIM : " << std::fixed << (allocH_time_/(double)(num_iter_)).count() * 1000000 << " us\n";
    std::cout << "Time taken to allocD PIM : " << std::fixed << (allocD_time_/(double)(num_iter_)).count() * 1000000 << " us\n";
    std::cout << "Time taken to align data : " << std::fixed << (aligning_time_/(double)(num_iter_)).count() * 1000000 << " us\n";
    std::cout << "Time taken to copyH2D_time_ PIM : " << std::fixed << (copyH2D_time_/(double)(num_iter_)).count() * 1000000 << " us\n";
    std::cout << "Time taken to copyD2H_time_ PIM : " << std::fixed << (copyD2H_time_/(double)(num_iter_)).count() * 1000000 << " us\n";
    std::cout << "Time taken to pim execute operation : " << std::fixed << (pim_kernel_time_/(double)(num_iter_)).count() * 1000000 << " us\n";
    std::cout << "Time taken to deallocH PIM : " << std::fixed << (deallocH_time_/(double)(num_iter_)).count() * 1000000 << " us\n";
    std::cout << "Time taken to deallocD PIM : " << std::fixed << (deallocD_time_/(double)(num_iter_)).count() * 1000000 << " us\n";
    std::cout << "Time taken to execute operation : " << std::fixed << kernel_execution_time_.count() * 1000000 << " us\n\n";


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

