#include "FimRuntime.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "executor/FimExecutor.h"
#include "utility/fim_log.h"

namespace fim
{
namespace runtime
{
FimRuntime::FimRuntime(FimRuntimeType rt_type, FimPrecision precision) : rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";

    fim_manager_ = fim::runtime::manager::FimManager::get_instance(rt_type, precision);
    fim_executor_ = fim::runtime::executor::FimExecutor::get_instance(rt_type, precision);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
}

int FimRuntime::initialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    fim_manager_->initialize();
    fim_executor_->initialize();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::deinitialize(void)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    fim_manager_->deinitialize();
    fim_executor_->deinitialize();

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::alloc_memory(void** ptr, size_t size, FimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->alloc_memory(ptr, size, mem_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::alloc_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->alloc_memory(fim_bo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::free_memory(void* ptr, FimMemType mem_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->free_memory(ptr, mem_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::free_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->free_memory(fim_bo);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::convert_data_layout(void* dst, void* src, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->convert_data_layout(dst, src, size, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::convert_data_layout(FimBo* dst, FimBo* src, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->convert_data_layout(dst, src, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::convert_data_layout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->convert_data_layout(dst, src0, src1, op_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::copy_memory(void* dst, void* src, size_t size, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->copy_memory(dst, src, size, cpy_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::copy_memory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_manager_->copy_memory(dst, src, cpy_type);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::execute_add(FimBo* output, FimBo* operand0, FimBo* operand1)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_executor_->execute_add(output, operand0, operand1);

    return ret;
}

int FimRuntime::execute_mul(FimBo* output, FimBo* operand0, FimBo* operand1)
{
    DLOG(INFO) << "called";
    int ret = 0;

    ret = fim_executor_->execute_mul(output, operand0, operand1);

    return ret;
}

int FimRuntime::execute_gemv(FimBo* output, FimBo* operand0, FimBo* operand1)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_executor_->execute_gemv(output, operand0, operand1);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::execute_relu(FimBo* output, FimBo* fim_data)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_executor_->execute_relu(output, fim_data);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

int FimRuntime::execute_bn(FimBo* output, FimBo* fim_data, FimBo* beta, FimBo* gamma, FimBo* mean, FimBo* variance,
                           double epsilon)
{
    DLOG(INFO) << "[START] " << __FUNCTION__ << " called";
    int ret = 0;

    ret = fim_executor_->execute_bn(output, fim_data, beta, gamma, mean, variance, epsilon);

    DLOG(INFO) << "[END] " << __FUNCTION__ << " called";
    return ret;
}

} /* namespace runtime */
} /* namespace fim */
