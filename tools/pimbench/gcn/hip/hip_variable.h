#ifndef CUDA_VARIABLE_CUH
#define CUDA_VARIABLE_CUH

#include "half.hpp"
#include "hip_kernel.h"
#include "sparse.h"

using half_float::half;
struct CUDAVariable {
public:
    float *data, *grad;
    bool requires_grad;
    int size;
    CUDAVariable(int size, bool requires_grad=true);
    ~CUDAVariable();

    void glorot(int in_size, int out_size);
    void zero();
    void zero_grad();
    void print(int col);
    void write2txt(std::string name, int col);
    void write2bin(std::string name);
    void readbin(std::string name);
    void readbin(std::string name, int cut_col, int origin_col);
    float grad_norm();
};

struct CUDASparseIndex {
public:
    int *indices, *indptr;
    int indices_size, indptr_size;

    CUDASparseIndex(): indices(nullptr), indptr(nullptr), indices_size(0), indptr_size(0) {}
    CUDASparseIndex(const SparseIndex &sp);
    ~CUDASparseIndex();
};

#endif
