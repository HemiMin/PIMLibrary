/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _FIM_EMULATOR_H_
#define _FIM_EMULATOR_H_

#include "FimTraceCoalescer.h"
#include "dramsim2/FimSimulator.h"
#include "fim_data_types.h"
#include "manager/FimInfo.h"

namespace fim
{
namespace runtime
{
namespace emulator
{
class FimEmulator
{
   public:
    FimEmulator();
    virtual ~FimEmulator(void) {}

    static FimEmulator* get_instance(void);

    int initialize(void);
    int deinitialize(void);
    int convert_mem_trace_from_16B_to_32B(FimMemTraceData* fmtd32, int* fmtd32_size, FimMemTraceData* fmtd16,
                                          int fmtd16_size, FimOpType op_type);
    int execute_bn(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size, uint64_t fim_base_addr,
                   uint8_t* temp_buf);
    int execute_elt_op(FimBo* output, FimBo* operand0, FimBo* operand1, FimMemTraceData* fmtd32, int fmtd32_size,
                       uint64_t fim_base_addr);
    int execute_relu(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size, uint64_t fim_base_addr);
    int execute_gemv(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size, FimOpType op_type,
                     uint64_t fim_base_addr, uint8_t* temp_buf);
    int execute_gemv_add(FimBo* output, FimBo* fim_data, FimMemTraceData* fmtd32, int fmtd32_size, FimOpType op_type,
                         uint64_t fim_base_addr, uint8_t* temp_buf);

   private:
    FimBlockInfo fbi_;
    FimSimulator fim_sim_;
    char* rocm_path;
};

} /* namespace emulator */
} /* namespace runtime */
} /* namespace fim */

#endif /* _FIM_EMULATOR_H_ */
