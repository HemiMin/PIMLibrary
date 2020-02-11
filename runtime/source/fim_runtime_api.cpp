#include "fim_runtime_api.h"
#include <iostream>
#include "FimRuntime.h"
#include "hip/hip_runtime.h"
#include "utility/fim_log.h"
#include "utility/fim_profile.h"

using namespace fim::runtime;

FimRuntime* fimRuntime = nullptr;
static bool log_initialized = false;

int FimInitialize(FimRuntimeType rtType, FimPrecision precision)
{
    if (!log_initialized) {
        google::InitGoogleLogging("FIMRuntime");
        FLAGS_minloglevel = FIM_LOG_LEVEL;
        log_initialized = true;
    }

    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Initialize);
    int ret = 0;

    if (fimRuntime == nullptr) fimRuntime = new FimRuntime(rtType, precision);
    ret = fimRuntime->Initialize();
    FIM_PROFILE_TOCK(Initialize);

    return ret;
}

int FimDeinitialize(void)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Deinitialize);
    int ret = 0;

    if (fimRuntime != nullptr) {
        ret = fimRuntime->Deinitialize();
        delete fimRuntime;
        fimRuntime = nullptr;
    }
    FIM_PROFILE_TOCK(Deinitialize);

    return ret;
}

int FimAllocMemory(void** ptr, size_t size, FimMemType memType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(AllocMemory);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->AllocMemory(ptr, size, memType);
    FIM_PROFILE_TOCK(AllocMemory);

    if (ptr == 0) return -1;

    return ret;
}

int FimAllocMemory(FimBo* fimBo)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(AllocMemory);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->AllocMemory(fimBo);
    FIM_PROFILE_TOCK(AllocMemory);

    if (fimBo->data == 0) return -1;

    return ret;
}

int FimFreeMemory(void* ptr, FimMemType memType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(FreeMemory);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->FreeMemory(ptr, memType);
    FIM_PROFILE_TOCK(FreeMemory);

    return ret;
}

int FimFreeMemory(FimBo* fimBo)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(FreeMemory);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->FreeMemory(fimBo);
    FIM_PROFILE_TOCK(FreeMemory);

    return ret;
}

int FimConvertDataLayout(void* dst, void* src, size_t size, FimOpType opType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->ConvertDataLayout(dst, src, size, opType);
    FIM_PROFILE_TOCK(ConvertDataLayout);

    return ret;
}

int FimConvertDataLayout(FimBo* dst, FimBo* src, FimOpType opType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->ConvertDataLayout(dst, src, opType);
    FIM_PROFILE_TOCK(ConvertDataLayout);

    return ret;
}

int FimConvertDataLayout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType opType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(ConvertDataLayout);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->ConvertDataLayout(dst, src0, src1, opType);
    FIM_PROFILE_TOCK(ConvertDataLayout);

    return ret;
}

int FimCopyMemory(void* dst, void* src, size_t size, FimMemcpyType cpyType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(CopyMemory);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->CopyMemory(dst, src, size, cpyType);
    FIM_PROFILE_TOCK(CopyMemory);

    return ret;
}

int FimCopyMemory(FimBo* dst, FimBo* src, FimMemcpyType cpyType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(CopyMemory);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->CopyMemory(dst, src, cpyType);
    FIM_PROFILE_TOCK(CopyMemory);

    return ret;
}

int FimExecute(void* output, void* operand0, void* operand1, size_t size, FimOpType opType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Execute);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->Execute(output, operand0, operand1, size, opType);
    FIM_PROFILE_TOCK(Execute);

    return ret;
}

int FimExecute(FimBo* output, FimBo* operand0, FimBo* operand1, FimOpType opType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Execute);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->Execute(output, operand0, operand1, opType);
    FIM_PROFILE_TOCK(Execute);

    return ret;
}

int FimExecute(FimBo* output, FimBo* fimData, FimOpType opType)
{
    DLOG(INFO) << "called";
    FIM_PROFILE_TICK(Execute);
    int ret = 0;

    if (fimRuntime == nullptr) {
        return -1;
    }
    ret = fimRuntime->Execute(output, fimData, opType);
    FIM_PROFILE_TOCK(Execute);

    return ret;
}
