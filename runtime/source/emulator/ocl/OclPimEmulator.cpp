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

#include "emulator/ocl/OclPimEmulator.h"
#include <CL/opencl.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "half.hpp"
#include "manager/ocl/OclMemoryManager.h"
#if AMD
#include "hip/hip_runtime.h"
#endif
#include "utility/assert_cl.h"
#include "utility/pim_debug.hpp"
#include "utility/pim_log.h"
#include "utility/pim_util.h"

namespace pim
{
namespace runtime
{
extern cl_command_queue queue;

namespace emulator
{
OclPimEmulator::OclPimEmulator(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";
    get_pim_block_info(&fbi_);
    this->initialize();
}

int OclPimEmulator::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " Intialization done ";
    int ret = 0;
#if AMD
    std::string rocm_path = ROCM_PATH;
    pim_sim_.initialize(rocm_path + "/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        rocm_path + "/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#else
    std::string pim_path = PIM_PATH;
    pim_sim_.initialize("/include/dramsim2/ini/HBM2_samsung_2M_16B_x64.ini",
                        "/include/dramsim2/ini/system_hbm_vega20.ini", 256 * 64 * 2, 64, 1);
#endif
    return ret;
}

int OclPimEmulator::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    return ret;
}

int OclPimEmulator::convert_mem_trace_from_16B_to_32B(PimMemTraceData* fmtd32, int* fmtd32_size,
                                                      PimMemTraceData* fmtd16, int fmtd16_size, PimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    DLOG(INFO) << "fmtd16_size : " << fmtd16_size;
    int ret = 0;
    TraceParser trace_converter;
    trace_converter.coalesce_trace(fmtd32, fmtd32_size, fmtd16, fmtd16_size);
#ifdef DEBUG_PIM
    const char* op_str = get_pim_op_string(op_type);
    std::string dump_data = TEST_VECTORS_DATA;
    dump_data.append("dump/");
    dump_data.append(op_str);
    std::string dump_fmtd16 = dump_data + "/fmtd16.dat";
    std::string dump_fmtd32 = dump_data + "/fmtd32.dat";
    dump_fmtd<16>(dump_fmtd16.c_str(), fmtd16, fmtd16_size);
    dump_fmtd<32>(dump_fmtd32.c_str(), fmtd32, fmtd32_size[0]);
#endif

    return ret;
}

int OclPimEmulator::execute_gemm_bias_act(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                                          PimOpType op_type, uint64_t pim_base_addr, uint8_t* temp_buf, PimBo* bias,
                                          PimActFunc act_func)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    int offset = 0;
    int offset_r = 0;

    int is_bias = (bias != nullptr) ? 1 : 0;
    int is_relu = (act_func == ACT_RELU) ? 1 : 0;

    uint16_t* sim_output = nullptr;
    uint16_t* h_bias = nullptr;
    int out_dim = output->bshape.n * output->bshape.c * output->bshape.h * output->bshape.w;
    sim_output = new uint16_t[out_dim];
    h_bias = new uint16_t[out_dim];
    uint64_t tmp_data_addr, pim_data_addr;
    void* input_data;

    tmp_data_addr = ((manager::OclBufferObj*)temp_buf)->host_addr;
    pim_data_addr = ((manager::OclBufferObj*)pim_data->data)->host_addr;
    input_data = (void*)(((manager::OclBufferObj*)pim_data->data)->host_addr);

    pim_sim_.preload_data_with_addr(pim_data_addr - pim_base_addr, input_data, pim_data->size);
    pim_sim_.execute_kernel((void*)fmtd32, fmtd32_size);
    pim_sim_.read_result_gemv(sim_output, tmp_data_addr - pim_base_addr, out_dim);

    if (is_bias) {
        clEnqueueReadBuffer(queue, (cl_mem)bias->data, CL_TRUE, 0, bias->size, (void*)h_bias, 0, NULL, NULL);
        for (int i = 0; i < out_dim; i++) {
            ((half_float::half*)sim_output)[i] += ((half_float::half*)h_bias)[i];
        }
    }

    if (is_relu) {
        for (int i = 0; i < out_dim; i++) {
            if (((half_float::half*)sim_output)[i] < 0) ((half_float::half*)sim_output)[i] = 0;
        }
    }

    if (output->mem_type != MEM_TYPE_HOST) {
        cl_int err_ret;
        void* host_addr = clEnqueueMapBuffer(queue, (cl_mem)output->data, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                             output->size, 0, NULL, NULL, &err_ret);
        cl_ok(err_ret);
        for (int n = 0; n < output->bshape.n; n++) {
            for (int c = 0; c < output->bshape.c; c++) {
                offset = n * output->bshape.c * output->bshape.h * output->bshape.w;
                offset += c * output->bshape.h * output->bshape.w;
                offset_r = n * output->bshape_r.c * output->bshape_r.h * output->bshape_r.w;
                offset_r += c * output->bshape_r.h * output->bshape_r.w;
                memcpy((half_float::half*)host_addr + offset_r, (half_float::half*)sim_output + offset,
                       pim_data->bshape_r.w * output->bshape_r.h * sizeof(half_float::half));
            }
        }
        cl_ok(clEnqueueUnmapMemObject(queue, (cl_mem)output->data, host_addr, 0, NULL, NULL));
        clFinish(queue);
    }

    delete[] sim_output;
    delete[] h_bias;

    return ret;
}

int OclPimEmulator::execute_gemv_add_tile_accum(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32,
                                                int fmtd32_size, PimOpType op_type, uint64_t pim_base_addr,
                                                uint8_t* temp_buf)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    uint16_t* sim_output = nullptr;
    int num_batch = output->bshape.n;
    int out_num = pim_data->bshape.h;
    int out_dim = out_num * num_batch;
    sim_output = new uint16_t[out_dim];
    uint64_t tmp_data_addr = reinterpret_cast<uint64_t>(temp_buf);
    uint64_t pim_data_addr = reinterpret_cast<uint64_t>(pim_data->data);
    int out_num_r = pim_data->bshape_r.h;
    int out_size_r = out_num_r * num_batch * sizeof(uint16_t);
    void* output_host = malloc(out_size_r);

    pim_sim_.preload_data_with_addr(pim_data_addr - pim_base_addr, pim_data->data, pim_data->size);
    pim_sim_.execute_kernel((void*)fmtd32, fmtd32_size);
    pim_sim_.read_result_gemv(sim_output, tmp_data_addr - pim_base_addr, out_dim);

    clEnqueueReadBuffer(queue, (cl_mem)output->data, CL_TRUE, 0, out_size_r, (void*)output_host, 0, NULL, NULL);
    for (int b = 0; b < num_batch; b++) {
        for (int i = 0; i < out_num_r; i++) {
            (reinterpret_cast<half_float::half*>(output_host))[b * out_num_r + i] +=
                (reinterpret_cast<half_float::half*>(sim_output))[b * out_num + i];
        }
    }
    clEnqueueWriteBuffer(queue, (cl_mem)output->data, CL_TRUE, 0, out_size_r, (void*)output_host, 0, NULL, NULL);

    free(output_host);
    delete[] sim_output;

    return ret;
}

int OclPimEmulator::execute_elt_op(PimBo* output, PimBo* operand0, PimBo* operand1, PimMemTraceData* fmtd32,
                                   int fmtd32_size, uint64_t pim_base_addr)
{
    DLOG(INFO) << "called";
    int ret = 0;
    int num_element = 0;
    uint16_t* sim_output = nullptr;
    num_element = output->size / sizeof(uint16_t);
    sim_output = new uint16_t[num_element];

    uint64_t input_addr[2], output_addr;
    void* input_data[2];

    if (operand0->mem_type == MEM_TYPE_PIM) {
        input_addr[0] = ((manager::OclBufferObj*)operand0->data)->host_addr;
    } else {
        input_addr[0] = reinterpret_cast<uint64_t>(operand0->data);
    }
    if (operand1->mem_type == MEM_TYPE_PIM) {
        input_addr[1] = ((manager::OclBufferObj*)operand1->data)->host_addr;
    } else {
        input_addr[1] = reinterpret_cast<uint64_t>(operand1->data);
    }
    if (output->mem_type == MEM_TYPE_PIM) {
        output_addr = ((manager::OclBufferObj*)output->data)->host_addr;
    } else {
        output_addr = reinterpret_cast<uint64_t>(output->data);
    }

    input_data[0] = (void*)(((manager::OclBufferObj*)operand0->data)->host_addr);
    input_data[1] = (void*)(((manager::OclBufferObj*)operand1->data)->host_addr);

    pim_sim_.preload_data_with_addr(input_addr[0] - pim_base_addr, input_data[0], operand0->size);
    pim_sim_.preload_data_with_addr(input_addr[1] - pim_base_addr, input_data[1], operand1->size);
    pim_sim_.execute_kernel((void*)fmtd32, (size_t)fmtd32_size);
    pim_sim_.read_result(sim_output, output_addr - pim_base_addr, output->size);

    memcpy((void*)output_addr, sim_output, output->size);

    delete[] sim_output;

    return ret;
}

int OclPimEmulator::execute_relu_bn_copy(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                                         uint64_t pim_base_addr)
{
    DLOG(INFO) << "called";
    int ret = 0;
    int num_element = 0;
    uint16_t* sim_output = nullptr;
    num_element = output->size / sizeof(uint16_t);
    sim_output = new uint16_t[num_element];

    uint64_t input_addr, output_addr;
    void* input_data;

    if (pim_data->mem_type == MEM_TYPE_PIM) {
        input_addr = ((manager::OclBufferObj*)pim_data->data)->host_addr;
    } else {
        input_addr = reinterpret_cast<uint64_t>(pim_data->data);
    }
    if (output->mem_type == MEM_TYPE_PIM) {
        output_addr = ((manager::OclBufferObj*)output->data)->host_addr;
    } else {
        output_addr = reinterpret_cast<uint64_t>(output->data);
    }

    input_data = (void*)(((manager::OclBufferObj*)pim_data->data)->host_addr);

    pim_sim_.preload_data_with_addr(input_addr - pim_base_addr, input_data, pim_data->size);
    pim_sim_.execute_kernel((void*)fmtd32, (size_t)fmtd32_size);
    pim_sim_.read_result(sim_output, output_addr - pim_base_addr, output->size);

    memcpy((void*)output_addr, sim_output, output->size);

    delete[] sim_output;

    return ret;
}

int OclPimEmulator::execute_bn(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                               uint64_t pim_base_addr, uint8_t* temp_buf)
{
    return execute_relu_bn_copy(output, pim_data, fmtd32, fmtd32_size, pim_base_addr);
}

int OclPimEmulator::execute_relu(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                                 uint64_t pim_base_addr)
{
    return execute_relu_bn_copy(output, pim_data, fmtd32, fmtd32_size, pim_base_addr);
}

int OclPimEmulator::execute_copy(PimBo* output, PimBo* pim_data, PimMemTraceData* fmtd32, int fmtd32_size,
                                 uint64_t pim_base_addr)
{
    return execute_relu_bn_copy(output, pim_data, fmtd32, fmtd32_size, pim_base_addr);
}
} /* namespace emulator */
} /* namespace runtime */
} /* namespace pim */
