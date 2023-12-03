#!/bin/bash
sed -i 's/rocrand.h/rocrand\/rocrand.h/g' /opt/rocm-4.0.0/include/hiprand/hiprand_hcc.h
sed -i 's/hiprand.h/hiprand\/hiprand.h/g' /opt/rocm-4.0.0/include/hiprand/hiprand_kernel.h
sed -i 's/rocrand_kernel.h/rocrand\/rocrand_kernel.h/g' /opt/rocm-4.0.0/include/hiprand/hiprand_kernel_hcc.h
