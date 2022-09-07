/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _OPENCL_MEM_MANAGER_H_
#define _OPENCL_MEM_MANAGER_H_

#include <CL/cl.h>
#include "manager/IPimMemoryManager.h"
#include "manager/PimDevice.h"
#include "manager/PimInfo.h"
#include "manager/ocl/OclBlockAllocator.h"
#include "manager/simple_heap.hpp"
#include "pim_data_types.h"

namespace pim
{
namespace runtime
{
namespace manager
{
class OclMemoryManager : public IPimMemoryManager
{
   public:
    OclMemoryManager(std::shared_ptr<PimDevice> pim_device, PimPrecision precision);
    virtual ~OclMemoryManager(void);

    int initialize(void);
    int deinitialize(void);
    int alloc_memory(void** ptr, size_t size, PimMemType mem_type);
    int alloc_memory(PimBo* pim_bo);
    int free_memory(void* ptr, PimMemType mem_type);
    int free_memory(PimBo* pim_bo);
    int copy_memory(void* dst, void* src, size_t size, PimMemCpyType cpy_type);
    int copy_memory(PimBo* dst, PimBo* src, PimMemCpyType cpy_type);
    int copy_memory_3d(const PimCopy3D* copy_params);
    int get_physical_id(void);
    int convert_data_layout(PimBo* dst, PimBo* src, PimOpType op_type) { return -1; };

   private:
    std::vector<SimpleHeap<OclBlockAllocator>*> fragment_allocator_;
    int host_id_;
    std::shared_ptr<PimDevice> pim_device_;
    PimPrecision precision_;
    PimBlockInfo* pbi_;

    cl_platform_id platform_;
    cl_device_id device_id_;
    cl_command_queue queue_;
    cl_context context_;
    cl_uint num_gpu_devices_;
};
}  // namespace manager
}  // namespace runtime
}  // namespace pim

#endif /*_OPENCL_MEM_MANAGER_H_ */
