/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */
#include "gemm_perf.h"
#include "utility/pim_profile.h"

using half_float::half;
using namespace std;

PimGemmTest::PimGemmTest(unsigned n, unsigned c, unsigned in_h, unsigned in_w, unsigned out_h, unsigned out_w,
                         PimActFunc act, bool has_bias, PimGemmOrder gemm_order, PerformanceAnalyser* pa)
    : n_(n),
      c_(c),
      in_h_(in_h),
      in_w_(in_w),
      out_h_(out_h),
      out_w_(out_w),
      act_(act),
      has_bias_(has_bias),
      gemm_order_(gemm_order), pa_(pa)
{
    if (!is_support_activation(act_)) {
        throw invalid_argument("Invalid activation type");
    }

    in_size_ = n_ * c_ * in_h_ * in_w_;
    out_size_ = n_ * c_ * out_h_ * out_w_;
    flt_ops_ = out_h_ * out_w_;
    if (gemm_order_ == W_X_I) {
        wgt_size_ = n_ * c_ * in_h_ * out_h_;
        flt_ops_ *= (2 * in_h_ - 1);
    } else {
        wgt_size_ = n_ * c_ * in_w_ * out_w_;
        flt_ops_ *= (2 * in_w_ - 1);
    }

    desc_ = PimCreateGemmDesc(n_, c_, in_h_, in_w_, out_h_, out_w_, PIM_FP16, gemm_order);

    pa_->Tick();
    PIM_PROFILE_TICK_A(PimAllocH1);
    h_i_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_INPUT);
    h_w_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_WEIGHT);
    h_o_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_OUTPUT);
    if (has_bias_) h_b_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_BIAS);
    PIM_PROFILE_TOCK_A(PimAllocH1);
    pa_->Tock();
    std::chrono::duration<double> allocH_time = pa->calculate_elapsed_time();
    pa_->accumulate_allocH_time(allocH_time);
    pa_->accumulate_total_time(allocH_time);
    std::cout << "allocH time1: " << allocH_time.count() * 1000 << std::endl << std::endl;

    pa_->Tick();
    PIM_PROFILE_TICK_A(PimAllocD1);
    d_i_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_INPUT);
    d_w_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_WEIGHT);
    d_o_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_OUTPUT);
    if (has_bias_) d_b_ = PimCreateBo(desc_, MEM_TYPE_DEVICE, GEMM_BIAS);
    PIM_PROFILE_TOCK_A(PimAllocD1);
    pa_->Tock();
    std::chrono::duration<double> allocD_time = pa_->calculate_elapsed_time();
    pa_->accumulate_allocD_time(allocD_time);
    pa_->accumulate_total_time(allocD_time);
    std::cout << "allocD time1: " << allocD_time.count() * 1000 << std::endl << std::endl;

    golden_ = PimCreateBo(desc_, MEM_TYPE_HOST, GEMM_OUTPUT);
}

PimGemmTest::~PimGemmTest()
{
    pa_->Tick();
    PIM_PROFILE_TICK_A(PimDeallocH);
    PimDestroyBo(h_i_);
    PimDestroyBo(h_w_);
    PimDestroyBo(h_o_);
    if (has_bias_) PimDestroyBo(h_b_);
    PIM_PROFILE_TOCK_A(PimDeallocH);
    pa_->Tock();
    std::chrono::duration<double> deallocH_time = pa_->calculate_elapsed_time();
    pa_->accumulate_deallocH_time(deallocH_time);
    pa_->accumulate_total_time(deallocH_time);
    std::cout << "deallocH time: " << deallocH_time.count() * 1000 << std::endl << std::endl;

    pa_->Tick();
    PIM_PROFILE_TICK_A(PimDeallocD);
    PimDestroyBo(d_i_);
    PimDestroyBo(d_w_);
    PimDestroyBo(d_o_);
    if (has_bias_) PimDestroyBo(d_b_);
    PIM_PROFILE_TOCK_A(PimDeallocD);
    pa_->Tock();
    std::chrono::duration<double> deallocD_time = pa_->calculate_elapsed_time();
    pa_->accumulate_deallocD_time(deallocD_time);
    pa_->accumulate_total_time(deallocD_time);
    std::cout << "deallocD time: " << deallocD_time.count() * 1000 << std::endl << std::endl;

    PimDestroyGemmDesc(desc_);

    PimDestroyBo(golden_);
}

void PimGemmTest::prepare(float alpha, float beta, float variation)
{
    set_half_data((half*)golden_->data, half(0.0), out_size_);
    set_half_data((half*)h_o_->data, half(0.0), out_size_);
    set_rand_half_data((half*)h_i_->data, half(variation), in_size_);
    set_rand_half_data((half*)h_w_->data, half(variation), wgt_size_);
    if (has_bias_) set_rand_half_data((half*)h_b_->data, half(variation), out_size_);

    half* h_i_data = (half*)h_i_->data;
    half* h_w_data = (half*)h_w_->data;
    half* golden_data = (half*)golden_->data;

    if (gemm_order_ == W_X_I) {
        for (int nc_i = 0; nc_i < n_ * c_; nc_i++) {
            matmulCPU(h_w_data, h_i_data, golden_data, out_h_, out_w_, out_w_, half(alpha), half(beta));
            h_i_data += (in_h_ * in_w_);
            h_w_data += (in_h_ * out_h_);
            golden_data += (out_h_ * out_w_);
        }
    } else {
        for (int nc_i = 0; nc_i < n_ * c_; nc_i++) {
            matmulCPU(h_i_data, h_w_data, golden_data, in_h_, out_w_, in_w_, half(alpha), half(beta));
            h_i_data += (in_h_ * in_w_);
            h_w_data += (in_w_ * out_w_);
            golden_data += (out_h_ * out_w_);
        }
    }
    if (has_bias_) {
        addBiasCPU((half*)golden_->data, (half*)h_b_->data, out_size_);
    }
    if (act_ == ACT_RELU) {
        reluCPU((half*)golden_->data, out_size_);
    }
    pa_->Tick();
    PIM_PROFILE_TICK_A(PimCopyH2D1);
    PimCopyMemory(d_i_, h_i_, HOST_TO_DEVICE);
    PimCopyMemory(d_w_, h_w_, HOST_TO_DEVICE);
    if (has_bias_) {
        PimCopyMemory(d_b_, h_b_, HOST_TO_DEVICE);
    } else {
        d_b_ = nullptr;
    }
    PIM_PROFILE_TOCK_A(PimCopyH2D1);
    pa_->Tock();
    std::chrono::duration<double> copyH2D_time = pa_->calculate_elapsed_time();
    pa_->accumulate_copyH2D_time(copyH2D_time);
    pa_->accumulate_total_time(copyH2D_time);
    std::cout << "copyH2D time1: " << copyH2D_time.count() * 1000 << std::endl << std::endl;

    PimCopyMemory(d_o_, h_o_, HOST_TO_DEVICE);
}

void PimGemmTest::execute_op(bool block)
{
    (void)PimExecuteGemm(d_o_, d_i_, d_w_, d_b_, act_, gemm_order_, nullptr, block);
    if (!block) {
      PimSynchronize();
    }
}

void PimGemmTest::finalize() 
{ 
  pa_->Tick();
  PIM_PROFILE_TICK_A(PimCopyD2H);
  PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST); 
  PIM_PROFILE_TOCK_A(PimCopyD2H);
  pa_->Tock();
  std::chrono::duration<double> copyD2H_time = pa_->calculate_elapsed_time();
  std::cout << "copyD2H time: " << copyD2H_time.count()*1000 << std::endl;
  pa_->accumulate_copyD2H_time(copyD2H_time);
}
void PimGemmTest::run_with_explicit_reordering(bool use_device_weight, bool block, unsigned niter)
{
    auto* w_to_reorder = use_device_weight ? d_w_ : h_w_;
    for (unsigned i = 0; i < niter; ++i) {
        auto* reordered_pim_w = PimConvertGemmWeight(w_to_reorder, gemm_order_);
        // Ignore return value here to avoid extra branches.
        // Please check the success of the API call in logs.
        // Results are verified further.
        (void)PimExecuteGemm(d_o_, d_i_, reordered_pim_w, d_b_, act_, gemm_order_, nullptr, block);
        if (!block) PimSynchronize();
        PimDestroyBo(reordered_pim_w);
    }
    PimCopyMemory(h_o_, d_o_, DEVICE_TO_HOST);
}

int PimGemmTest::validate(float epsilon)
{
    return compare_half_relative((half*)h_o_->data, (half*)golden_->data, out_size_, epsilon);
}

double PimGemmTest::get_flt_ops() { return flt_ops_; }
PimGemmTestFixture::PimGemmTestFixture() {}
int PimGemmTestFixture::ExecuteTest()
{
    act = (parser_->get_act_function() == "relu") ? ACT_RELU : NONE;
    has_bias = (parser_->get_has_bias()) ? true : false;
    PimGemmTest pimGemmTest = PimGemmTest(num_batch_, num_channels_, input_height_, input_width_, output_height_,
                                          output_width_, act, has_bias, order_, (PerformanceAnalyser*)this);
    pimGemmTest.prepare();

    // warmup
    pimGemmTest.execute_op(true);
    std::cout << "=======WarmUp end========" << std::endl;

    for (int i = 0; i < num_iter_; i++) {
        Tick();
        PIM_PROFILE_TICK_A(PimExecuteGemm);
        pimGemmTest.execute_op(block_);
        PIM_PROFILE_TOCK_A(PimExecuteGemm);
        Tock();
        std::chrono::duration<double> pim_time = calculate_elapsed_time();
        accumulate_pim_kernel_time(pim_time);
        std::cout << "pimExecuteGemm time: " << pim_time.count() * 1000 << std::endl << std::endl;
        //Tock();
        //avg_kernel_time_ += calculate_elapsed_time();
    }

    auto start = std::chrono::high_resolution_clock::now();
    pimGemmTest.finalize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end - start;
    accumulate_total_time(total_time);
    accumulate_total_time(pim_kernel_time_/(double)num_iter_);

    kernel_execution_time_ = avg_kernel_time_;
    //calculate_avg_time();
    calculate_gflops(pimGemmTest.get_flt_ops());
    return pimGemmTest.validate();
}

int PimGemmTestFixture::ExecuteTestExplicitReordering()
{
    bool use_device_weight = false;
    PimGemmTest pimGemmTest = PimGemmTest(num_batch_, num_channels_, input_height_, input_width_, output_height_,
                                          output_width_, act, has_bias, order_, (PerformanceAnalyser*)this);
    pimGemmTest.prepare();
    pimGemmTest.run_with_explicit_reordering(use_device_weight, block_);

    avg_kernel_time_ = std::chrono::duration<double>::zero();
    for (int i = 0; i < num_iter_; i++) {
        Tick();
        pimGemmTest.run_with_explicit_reordering(use_device_weight, block_);
        Tock();
        avg_kernel_time_ += calculate_elapsed_time();
    }
    pimGemmTest.finalize();
    calculate_avg_time();
    calculate_gflops(pimGemmTest.get_flt_ops());

    return pimGemmTest.validate();
}
