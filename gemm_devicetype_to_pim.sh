#!/bin/bash

sed -i 's/MEM_TYPE_DEVICE/MEM_TYPE_PIM/g' tools/pimbench/source/gemm_perf.cpp
sed -i 's/HOST_TO_DEVICE/HOST_TO_PIM/g' tools/pimbench/source/gemm_perf.cpp
sed -i 's/DEVICE_TO_HOST/PIM_TO_HOST/g' tools/pimbench/source/gemm_perf.cpp

