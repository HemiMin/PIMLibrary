#include "hip_variable.h"
#include <fstream>

CUDAVariable::CUDAVariable(int size, bool requires_grad) {
    this->requires_grad = requires_grad;
    this->size = size;
    CUDA_CHECK(hipMalloc((void**) &data, size * sizeof(float)));
    if (requires_grad) {
        CUDA_CHECK(hipMalloc((void**) &grad, size * sizeof(float)));
    }
}

CUDAVariable::~CUDAVariable() {
    CUDA_CHECK(hipFree(data));
    if (requires_grad) CUDA_CHECK(hipFree(grad));
}

void CUDAVariable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0f / (in_size + out_size)), scale = range * 2;

    dim3 block((size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    hipLaunchKernelGGL(cuda_Variable_glorot_kernel,block, thread_in_block,0,0,data, devStates, size, scale);
}

void CUDAVariable::zero() {
    CUDA_CHECK(hipMemset(data, 0, size * sizeof(float)));
}

void CUDAVariable::zero_grad() {
    CUDA_CHECK(hipMemset(grad, 0, size * sizeof(float)));
}

void CUDAVariable::print(int col) {
    float cpu_data[size];
    CUDA_CHECK(hipMemcpy(cpu_data, data, size * sizeof(float), hipMemcpyDeviceToHost));
    int count = 0;
    for (int i = 0; i < size; ++i) {
        printf("%.4f ", cpu_data[i]);
        count++;
        if (count % col == 0) printf("\n");
    }
    printf("\n");
}

void CUDAVariable::write2txt(std::string name, int col) {
  std::vector<float> cpu_data(size);
  std::ofstream out(name);
    CUDA_CHECK(hipMemcpy(cpu_data.data(), data, size * sizeof(float), hipMemcpyDeviceToHost));
    int count = 0;
    for (int i = 0; i < size; ++i) {
        out << cpu_data[i] << " ";
        count++;
        if (count % col == 0) out << "\n";
    }
}

void CUDAVariable::write2bin(std::string name) {
  std::vector<float> cpu_data(size);
  CUDA_CHECK(hipMemcpy(cpu_data.data(), data, size * sizeof(float), hipMemcpyDeviceToHost));
  std::ofstream out(name, std::ios::out | std::ios::binary);
  for (int i = 0; i < size; ++i) {
    out.write(reinterpret_cast<const char*>(&cpu_data[i]), sizeof(float));
  }
  out.close();
}

void CUDAVariable::readbin(std::string name) {
  std::ifstream in(name, std::ios::binary);
  int idx = 0;
  int r = 0;
  float x;
  std::vector<float> cpu_data;
  while(in.read(reinterpret_cast<char*>(&x), sizeof(float))){
      cpu_data.push_back(x);
      if (cpu_data.size() == x) break;
  }
  CUDA_CHECK(hipMemcpy(data, cpu_data.data(), size * sizeof(float), hipMemcpyHostToDevice));
}

void CUDAVariable::readbin(std::string name, int cut_col, int origin_col) {
  std::ifstream in(name, std::ios::binary);
  int idx = 0;
  int r = 0;
  int c = 0;
  float x;
  std::vector<float> cpu_data;
  while(in.read(reinterpret_cast<char*>(&x), sizeof(float))){
    if (c/origin_col>0) {
      c=0;
    }if (c/cut_col >0) {
      c++;
      continue;
    } else {
      c++;
      cpu_data.push_back(x);
      if (cpu_data.size() == x) break;
    }
  }
  CUDA_CHECK(hipMemcpy(data, cpu_data.data(), size * sizeof(float), hipMemcpyHostToDevice));
}

float CUDAVariable::grad_norm() {
    float norm = 0;
    float *cpu_grad = new float[size];
    CUDA_CHECK(hipMemcpy(cpu_grad, grad, size * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < size; ++i)
        norm += cpu_grad[i] * cpu_grad[i];
    delete[] cpu_grad;
    return sqrtf(norm);
}

CUDASparseIndex::CUDASparseIndex(const SparseIndex &sp) {
    indices_size = sp.indices.size();
    indptr_size = sp.indptr.size();

    CUDA_CHECK(hipMalloc((void**) &indices, indices_size * sizeof(int)));
    CUDA_CHECK(hipMalloc((void**) &indptr, indptr_size * sizeof(int)));

    CUDA_CHECK(hipMemcpy(indices, sp.indices.data(), indices_size * sizeof(int), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(indptr, sp.indptr.data(), indptr_size * sizeof(int), hipMemcpyHostToDevice));
}

CUDASparseIndex::~CUDASparseIndex() {
    if (indices != nullptr) CUDA_CHECK(hipFree(indices));
    if (indptr != nullptr) CUDA_CHECK(hipFree(indptr));
}
