/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _PIM_DATA_TYPE_H_
#define _PIM_DATA_TYPE_H_

#include <stddef.h>
#include <stdint.h>

#define __PIM_API__

typedef enum __PimRuntimeType {
    RT_TYPE_HIP,
    RT_TYPE_OPENCL,
} PimRuntimeType;
typedef enum __PimMemType {
    MEM_TYPE_HOST,
    MEM_TYPE_DEVICE,
    MEM_TYPE_PIM,
} PimMemType;

typedef enum __PimMemFlag { ELT_OP, GEMV_INPUT, GEMV_WEIGHT, GEMV_OUTPUT } PimMemFlag;

typedef enum __PimMemCpyType {
    HOST_TO_HOST,
    HOST_TO_DEVICE,
    HOST_TO_PIM,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE,
    DEVICE_TO_PIM,
    PIM_TO_HOST,
    PIM_TO_DEVICE,
    PIM_TO_PIM,
} PimMemCpyType;

typedef enum __PimOpType {
    OP_GEMV,
    OP_ELT_ADD,
    OP_ELT_MUL,
    OP_RELU,
    OP_BN,
    OP_DUMMY,
} PimOpType;

typedef enum __PimPrecision {
    PIM_FP16,
    PIM_INT8,
} PimPrecision;

typedef struct __PimBShape {
    uint32_t w;
    uint32_t h;
    uint32_t c;
    uint32_t n;
} PimBShape;

typedef struct __PimBufferObject {
    PimMemType mem_type;
    PimBShape bshape;
    PimBShape bshape_r;
    PimPrecision precision;
    size_t size;
    void* data;
    bool use_user_ptr;
} PimBo;

typedef struct __PimDescriptor {
    PimBShape bshape;
    PimBShape bshape_r;
    PimPrecision precision;
    PimOpType op_type;
} PimDesc;

#endif /* _PIM_DATA_TYPE_H_ */
