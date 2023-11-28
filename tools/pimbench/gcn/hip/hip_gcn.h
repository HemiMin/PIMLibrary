#ifndef CUDA_GCN_CUH
#define CUDA_GCN_CUH

#include "gcn.h"
#include "hip_variable.h"
#include "hip_module.h"
#include "common_perf.h"

using std::vector;
using std::pair;

class CUDAGCN {
    vector<CUDAModule*> modules;
    vector<CUDAVariable> variables;
    CUDAVariable *input, *output;
    CUDASparseIndex *sp, *graph;
    CUDAAdam *optimizer;
    int *truth;
    float loss;
    float *d_l2_penalty;

    void set_input();
    void set_truth(int current_split);
    float get_accuracy();
    float get_l2_penalty();
    pair<float, float> train_epoch();
    pair<float, float> eval(int current_split);
    GCNData *data;
public:
    GCNParams params;
    CUDAGCN(GCNParams params, GCNData *input_data, PerformanceAnalyser* pa);
    CUDAGCN() {}
    ~CUDAGCN();
    void run();
};

#endif
