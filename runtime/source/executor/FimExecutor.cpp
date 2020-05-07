#include "executor/FimExecutor.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "executor/fim_hip_kernels/fim_op_kernels.fimk"
#include "hip/hip_runtime.h"
#include "utility/fim_dump.hpp"
#include "utility/fim_log.h"
#include "utility/fim_util.h"

namespace fim
{
namespace runtime
{
namespace executor
{
FimExecutor::FimExecutor(FimRuntimeType rt_type, FimPrecision precision)
    : rt_type_(rt_type), precision_(precision), thread_cnt_(16)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called ";

#ifdef EMULATOR
    fim_emulator_ = fim::runtime::emulator::FimEmulator::get_instance();

    get_fim_block_info(&fbi_);
    fmtd_size_per_ch_ = 4000;
    max_block_size_ = fbi_.num_fim_chan;
    max_fmtd_size_ = fmtd_size_per_ch_ * max_block_size_;
#endif
    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

FimExecutor* FimExecutor::get_instance(FimRuntimeType rt_type, FimPrecision precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " Called";
    static FimExecutor* instance_ = new FimExecutor(rt_type, precision);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return instance_;
}

int FimExecutor::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " Intialization done ";

    int ret = 0;
    hipGetDeviceProperties(&dev_prop_, 0);
    std::cout << " System minor " << dev_prop_.minor << std::endl;
    std::cout << " System major " << dev_prop_.major << std::endl;
    std::cout << " agent prop name " << dev_prop_.name << std::endl;
    std::cout << " hip Device prop succeeded " << std::endl;

    fim_manager_ = fim::runtime::manager::FimManager::get_instance(rt_type_, precision_);

    int max_crf_size = 128;
    int max_srf_size = 2048;
    hipMalloc((void**)&d_crf_bin_buffer_, max_crf_size);
    hipMalloc((void**)&d_srf_bin_buffer_, max_srf_size);

#ifdef EMULATOR
    int dummy_size = 20 * 1024 * 1024;
    int reserved_fmtd_size = max_fmtd_size_ * sizeof(FimMemTraceData);
    hipMalloc((void**)&fim_base_addr_, dummy_size);
    hipMalloc((void**)&d_fmtd16_, reserved_fmtd_size);
    hipMalloc((void**)&d_fmtd16_size_, sizeof(int));

    h_fmtd16_ = (FimMemTraceData*)malloc(reserved_fmtd_size);
    h_fmtd32_ = (FimMemTraceData*)malloc(reserved_fmtd_size);
    h_fmtd16_size_ = (int*)malloc(sizeof(int));
    h_fmtd32_size_ = (int*)malloc(sizeof(int));
#else
    /* TODO: get fim control base address from device driver */
    /* roct should write fim_base_va */
    FILE* fp;
    fp = fopen("fim_base_va.txt", "rt");
    fscanf(fp, "%lX", &fim_base_addr_);
    printf("fim_base_addr_ : 0x%lX\n", fim_base_addr_);
    fclose(fp);
    fmtd16_ = nullptr;
    fmtd32_ = nullptr;
    fmtd16_size_ = nullptr;
    fmtd32_size_ = nullptr;
#endif
    /* FIM HW can generate only gemv output without reduction sum */
    /* so FimExecutor needs to maintain intermediate output buffer for gemv op */
    hipMalloc((void**)&fim_gemv_tmp_buffer_, 2 * 1024 * 1024);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    hipFree((void*)d_crf_bin_buffer_);
    hipFree((void*)d_srf_bin_buffer_);
#ifdef EMULATOR
    hipFree((void*)fim_base_addr_);
    hipFree((void*)d_fmtd16_);
    hipFree((void*)d_fmtd16_size_);
    free(h_fmtd16_);
    free(h_fmtd16_size_);
    free(h_fmtd32_);
    free(h_fmtd32_size_);
#endif
    hipFree((void*)fim_gemv_tmp_buffer_);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute(void* output, void* operand0, void* operand1, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    if (op_type == OP_ELT_ADD) {
    } else {
        /* todo:implement other operation function */
        return -1;
    }
    hipStreamSynchronize(NULL);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute(FimBo* output, FimBo* operand0, FimBo* operand1, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    size_t size = output->size;
    FimBo* input = operand0;
    FimBo* weight = operand1;

    fim_manager_->create_crf_binary(op_type, input->size, output->size);
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();

    // FIXME : change 128 to a meaningful variable.
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);

    if (op_type == OP_GEMV) {
        hipMemcpy((void*)fim_base_addr_, weight->data, weight->size, hipMemcpyDeviceToDevice);
        hipLaunchKernelGGL(
            gemv_fim_1cu_2th_fp16, dim3(1), dim3(2), 0, 0, (uint8_t*)fim_base_addr_ /* fim control base */,
            (uint8_t*)fim_base_addr_ /* fim weight base */, (uint8_t*)fim_gemv_tmp_buffer_, /* fim hw output buffer */
            (uint8_t*)input->data, (uint8_t*)output->data, input->bshape.w, output->bshape.w,
            (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, (uint8_t*)d_crf_bin_buffer_, crf_size);
    } else {
        /* todo:implement other operation function */
        hipLaunchKernelGGL(dummy_kernel, dim3(1), dim3(1), 0, 0);
        return -1;
    }
    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    printf("%s %d d_fmtd16_size : %d\n", __func__, __LINE__, d_fmtd16_size_[0]);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    if (op_type == OP_GEMV) {
        fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0],
                                                         op_type);
        fim_emulator_->execute_fim(output, weight, h_fmtd32_, h_fmtd32_size_[0], op_type);
    }
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute(FimBo* output, FimBo* fim_data, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    unsigned blocks = max_block_size_;
    unsigned threads_per_block = 16;

    fim_manager_->create_crf_binary(op_type, output->size, output->size);
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();

    // FIXME : change 128 to a meaningful variable.
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);

    if (op_type == OP_ELT_ADD) {
        blocks = 1;
        threads_per_block = 2;
        hipMemcpy((void*)fim_base_addr_, fim_data->data, fim_data->size, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(elt_add_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0,
                           (uint8_t*)fim_base_addr_, (uint8_t*)fim_base_addr_, (uint8_t*)output->data,
                           (int)output->size, (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                           (uint8_t*)d_crf_bin_buffer_, crf_size);

    } else if (op_type == OP_ELT_MUL) {
        blocks = 1;
        threads_per_block = 2;
        hipMemcpy((void*)fim_base_addr_, fim_data->data, fim_data->size, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(elt_mul_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0,
                           (uint8_t*)fim_base_addr_, (uint8_t*)fim_base_addr_, (uint8_t*)output->data,
                           (int)output->size, (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                           (uint8_t*)d_crf_bin_buffer_, crf_size);
    } else if (op_type == OP_RELU) {
        blocks = 1;
        threads_per_block = 2;
        hipMemcpy((void*)fim_base_addr_, fim_data->data, fim_data->size, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(relu_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (uint8_t*)fim_base_addr_,
                           (uint8_t*)fim_base_addr_, (uint8_t*)output->data, (int)output->size,
                           (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_, fmtd_size_per_ch_,
                           (uint8_t*)d_crf_bin_buffer_, crf_size);
    } else {
        /* todo:implement other operation function */
        hipLaunchKernelGGL(dummy_kernel, dim3(1), dim3(1), 0, 0);
        DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
        return -1;
    }
    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    if (op_type == OP_ELT_ADD || op_type == OP_ELT_MUL || op_type == OP_RELU) {
        for (size_t i = 1; i < blocks; i++) {
            memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
                   h_fmtd16_size_[0] * sizeof(FimMemTraceData));
        }
        h_fmtd16_size_[0] *= blocks;
        fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0],
                                                         op_type);
        fim_emulator_->execute_fim(output, fim_data, h_fmtd32_, h_fmtd32_size_[0], op_type);
    }
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_add(FimBo* output, FimBo* fim_data)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    fim_manager_->create_crf_binary(OP_ELT_ADD, output->size, output->size);
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();

    // FIXME : change 128 to a meaningful variable.
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);
    hipMemcpy((void*)fim_base_addr_, fim_data->data, fim_data->size, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(elt_add_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (uint8_t*)fim_base_addr_,
                       (uint8_t*)fim_base_addr_, (uint8_t*)output->data, output->size, (FimMemTraceData*)d_fmtd16_,
                       (int*)d_fmtd16_size_, fmtd_size_per_ch_, (uint8_t*)d_crf_bin_buffer_, crf_size);

    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0],
                                                     OP_ELT_ADD);
    fim_emulator_->execute_fim(output, fim_data, h_fmtd32_, h_fmtd32_size_[0], OP_ELT_ADD);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_mul(FimBo* output, FimBo* fim_data)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    fim_manager_->create_crf_binary(OP_ELT_MUL, output->size, output->size);
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();

    // FIXME : change 128 to a meaningful variable.
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);
    hipMemcpy((void*)fim_base_addr_, fim_data->data, fim_data->size, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(elt_mul_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (uint8_t*)fim_base_addr_,
                       (uint8_t*)fim_base_addr_, (uint8_t*)output->data, (int)output->size, (FimMemTraceData*)d_fmtd16_,
                       (int*)d_fmtd16_size_, fmtd_size_per_ch_, (uint8_t*)d_crf_bin_buffer_, crf_size);

    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0],
                                                     OP_ELT_MUL);
    fim_emulator_->execute_fim(output, fim_data, h_fmtd32_, h_fmtd32_size_[0], OP_ELT_MUL);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_gemv(FimBo* output, FimBo* operand0, FimBo* operand1)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    FimBo* input = operand0;
    FimBo* weight = operand1;

    int in_size = weight->bshape.w;
    int out_size = weight->bshape.h;

    fim_manager_->create_crf_binary(OP_GEMV, in_size * sizeof(half), out_size * sizeof(half));
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();

    // FIXME : change 128 to a meaningful variable.
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);

    hipMemcpy((void*)fim_base_addr_, weight->data, weight->size, hipMemcpyDeviceToDevice);
    hipLaunchKernelGGL(gemv_fim_1cu_2th_fp16, dim3(1), dim3(2), 0, 0, (uint8_t*)fim_base_addr_ /* fim control base */,
                       (uint8_t*)fim_base_addr_ /* fim weight base */,
                       (uint8_t*)fim_gemv_tmp_buffer_, /* fim hw output buffer */
                       (uint8_t*)input->data, (uint8_t*)output->data, in_size, out_size, (FimMemTraceData*)d_fmtd16_,
                       (int*)d_fmtd16_size_, (uint8_t*)d_crf_bin_buffer_, crf_size);

    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    printf("%s %d d_fmtd16_size : %d\n", __func__, __LINE__, d_fmtd16_size_[0]);
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_GEMV);
    fim_emulator_->execute_fim(output, weight, h_fmtd32_, h_fmtd32_size_[0], OP_GEMV);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_relu(FimBo* output, FimBo* fim_data)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;
    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    fim_manager_->create_crf_binary(OP_RELU, output->size, output->size);
    uint8_t* crf_binary = fim_manager_->get_crf_binary();
    int crf_size = fim_manager_->get_crf_size();

    // FIXME : change 128 to a meaningful variable.
    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, sizeof(uint8_t) * 128, hipMemcpyHostToDevice);
    hipMemcpy((void*)fim_base_addr_, fim_data->data, fim_data->size, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(relu_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (uint8_t*)fim_base_addr_,
                       (uint8_t*)fim_base_addr_, (uint8_t*)output->data, (int)output->size, (FimMemTraceData*)d_fmtd16_,
                       (int*)d_fmtd16_size_, fmtd_size_per_ch_, (uint8_t*)d_crf_bin_buffer_, crf_size);
    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_RELU);
    fim_emulator_->execute_fim(output, fim_data, h_fmtd32_, h_fmtd32_size_[0], OP_RELU);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimExecutor::execute_bn(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* scale, FimBo* shift)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    unsigned blocks = 1;
    unsigned threads_per_block = 2;

    /* TODO: implement srf bin generator */
    uint8_t* crf_binary = bn_crf;
    uint8_t* srf_binary = bn_srf_data;
    int crf_size = sizeof(bn_crf);
    int srf_size = sizeof(bn_srf_data);

    hipMemcpy((void*)d_crf_bin_buffer_, (void*)crf_binary, crf_size, hipMemcpyHostToDevice);
    hipMemcpy((void*)d_srf_bin_buffer_, (void*)srf_binary, srf_size, hipMemcpyHostToDevice);
    hipMemcpy((void*)fim_base_addr_, fim_data->data, fim_data->size, hipMemcpyHostToDevice);

    printf("crf_size:%d, srf_size:%d, output->size:%d\n", crf_size, srf_size, output->size);
    printf("bshaped(%d,%d,%d,%d)\n", output->bshape.w, output->bshape.h, output->bshape.c, output->bshape.n);

    hipLaunchKernelGGL(bn_fim_1cu_2th_fp16, dim3(blocks), dim3(threads_per_block), 0, 0, (uint8_t*)fim_base_addr_,
                       (uint8_t*)fim_base_addr_, (uint8_t*)output->data, (int)output->size, output->bshape.n,
                       output->bshape.c, output->bshape.w, (uint8_t*)d_crf_bin_buffer_, crf_size,
                       (uint8_t*)d_srf_bin_buffer_, srf_size, (FimMemTraceData*)d_fmtd16_, (int*)d_fmtd16_size_,
                       fmtd_size_per_ch_);

    hipStreamSynchronize(NULL);

#ifdef EMULATOR
    hipMemcpy((void*)h_fmtd16_size_, (void*)d_fmtd16_size_, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy((void*)h_fmtd16_, (void*)d_fmtd16_, sizeof(FimMemTraceData) * max_fmtd_size_, hipMemcpyDeviceToHost);

    for (size_t i = 1; i < blocks; i++) {
        memcpy(&h_fmtd16_[i * h_fmtd16_size_[0]], &h_fmtd16_[i * fmtd_size_per_ch_],
               h_fmtd16_size_[0] * sizeof(FimMemTraceData));
    }
    h_fmtd16_size_[0] *= blocks;
    fim_emulator_->convert_mem_trace_from_16B_to_32B(h_fmtd32_, h_fmtd32_size_, h_fmtd16_, h_fmtd16_size_[0], OP_BN);
    fim_emulator_->execute_fim(output, fim_data, h_fmtd32_, h_fmtd32_size_[0], OP_BN);
#endif

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

} /* namespace executor */
} /* namespace runtime */
} /* namespace fim */
