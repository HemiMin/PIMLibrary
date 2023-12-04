#!/bin/bash

sed -i 's/MEM_TYPE_DEVICE/MEM_TYPE_PIM/g' tools/pimbench/gcn/hip/hip_gcn.cpp
sed -i 's/HOST_TO_DEVICE/HOST_TO_PIM/g' tools/pimbench/gcn/hip/hip_gcn.cpp
sed -i 's/DEVICE_TO_DEVICE/PIM_TO_PIM/g' tools/pimbench/gcn/hip/hip_gcn.cpp
sed -i 's/DEVICE_TO_HOST/PIM_TO_HOST/g' tools/pimbench/gcn/hip/hip_gcn.cpp

