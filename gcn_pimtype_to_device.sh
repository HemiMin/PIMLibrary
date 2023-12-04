#!/bin/bash

sed -i 's/MEM_TYPE_PIM/MEM_TYPE_DEVICE/g' tools/pimbench/gcn/hip/hip_gcn.cpp
sed -i 's/HOST_TO_PIM/HOST_TO_DEVICE/g' tools/pimbench/gcn/hip/hip_gcn.cpp
sed -i 's/PIM_TO_PIM/DEVICE_TO_DEVICE/g' tools/pimbench/gcn/hip/hip_gcn.cpp
sed -i 's/PIM_TO_HOST/DEVICE_TO_HOST/g' tools/pimbench/gcn/hip/hip_gcn.cpp

