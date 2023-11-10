#include "hip_module.h"
#include "timer.h"
#include "rand.h"
#include <rocblas.h>

CUDAMatmul::CUDAMatmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, int m, int n, int p) : 
    a(a), b(b), c(c), m(m), n(n), p(p) {}

void CUDAMatmul::forward(bool training) {
    timer_start(TMR_MATMUL_FW);

    //c->zero();
    //dim3 block((p-1) / TILE_SIZE + 1, (m-1) / TILE_SIZE + 1, 1);
    //dim3 thread_in_block(TILE_SIZE, TILE_SIZE, 1);
    //hipLaunchKernelGGL(cuda_Matmul_forward_kernel,block, thread_in_block,0,0,a->data, b->data, c->data, m, n, p);
    //CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    rocblas_status rstatus = rocblas_status_success;
    rocblas_int M = m;
    rocblas_int K = n;
    rocblas_int N = p;
    float alpha = 1.0;
    float beta = 0.0;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_int lda, ldb, ldc, sizeA, sizeB, sizeC;
    if(transA == rocblas_operation_none)
    { lda = M; sizeA = K * lda; }
    else { lda = K; sizeA = M * lda; }
    if(transB == rocblas_operation_none)
    { ldb = K; sizeB = N * ldb; }
    else { ldb = N; sizeB = K * ldb; }
    ldc = M; sizeC = N * ldc;

    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);
    rstatus = rocblas_sgemm(handle, transA, transB, N, M, K, &alpha, 
                            b->data,  N, a->data, K, &beta, c->data, N);
    CHECK_ROCBLAS_STATUS(rstatus);

    timer_stop(TMR_MATMUL_FW);
}

void CUDAMatmul::backward() {
    timer_start(TMR_MATMUL_BW);

    a->zero_grad();
    b->zero_grad();
    dim3 block_a((n-1)/TILE_SIZE+1, (m-1)/TILE_SIZE+1, 1);
    dim3 block_b((p-1)/TILE_SIZE+1, (n-1)/TILE_SIZE+1, 1);
    dim3 thread_in_block(TILE_SIZE, TILE_SIZE, 1);
    hipLaunchKernelGGL(cuda_Matmul_backward_A_kernel,block_a, thread_in_block,0,0,a->grad, b->data, c->grad, m, n, p);
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());
    cuda_Matmul_backward_B_kernel<<<block_b, thread_in_block>>>(b->grad, a->data, c->grad, m, n, p);
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_MATMUL_BW);
}

CUDASparseMatmul::CUDASparseMatmul(CUDAVariable *a, CUDAVariable *b, CUDAVariable *c, CUDASparseIndex *sp, int m, int n, int p) : 
    a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

void CUDASparseMatmul::forward(bool training) {
    timer_start(TMR_SPMATMUL_FW);

    c->zero();
    // TODO: when p larger than 1024?
    if (sp->indptr_size <= 1) return;

    dim3 block(sp->indptr_size - 1, 1, 1);
    dim3 thread_in_block(p, 1, 1);
    hipLaunchKernelGGL(cuda_SparseMatmul_forward_kernel,block, thread_in_block,0,0,a->data, b->data, c->data, sp->indptr, sp->indices, p);
    CUDA_CHECK(hipGetLastError());

    timer_stop(TMR_SPMATMUL_FW);
}

void CUDASparseMatmul::backward() {
    timer_start(TMR_SPMATMUL_BW);

    b->zero_grad();
    // TODO: when p larger than 1024?
    if (sp->indptr_size <= 1) return;
    dim3 block(sp->indptr_size - 1, 1, 1);
    dim3 thread_in_block(p, 1, 1);
    hipLaunchKernelGGL(cuda_SparseMatmul_backward_kernel,block, thread_in_block,0,0,a->data, b->grad, c->grad, sp->indptr, sp->indices, p);
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_SPMATMUL_BW);
}

CUDAGraphSum::CUDAGraphSum(CUDAVariable *in, CUDAVariable *out, CUDASparseIndex *graph, int dim) : 
    in(in), out(out), graph(graph), dim(dim) {}

void CUDAGraphSum::forward(bool training) {
    timer_start(TMR_GRAPHSUM_FW);

    out->zero();
    // TODO: when dim larger than 1024?
    const int numNodes = graph->indptr_size - 1;
    dim3 block(numNodes, 1, 1);
    dim3 thread_in_block(dim, 1, 1);
    hipLaunchKernelGGL(cuda_GraphSum_forward_kernel,block, thread_in_block,0,0,in->data, out->data, graph->indptr, graph->indices, dim, numNodes);
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_GRAPHSUM_FW);
}

void CUDAGraphSum::backward() {
    timer_start(TMR_GRAPHSUM_BW);

    in->zero_grad();
    // TODO: when dim larger than 1024?
    const int numNodes = graph->indptr_size - 1;
    dim3 block(numNodes, 1, 1);
    dim3 thread_in_block(dim, 1, 1);
    hipLaunchKernelGGL(cuda_GraphSum_backward_kernel,block, thread_in_block,0,0,in->grad, out->grad, graph->indptr, graph->indices, dim, numNodes);
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_GRAPHSUM_BW);
}

CUDACrossEntropyLoss::CUDACrossEntropyLoss(CUDAVariable *logits, int *truth, float *loss, int num_classes) :
    logits(logits), truth(truth), loss(loss), num_classes(num_classes) {
    int logitsPerClass = logits->size / num_classes;
    CUDA_CHECK(hipMalloc((void**) &d_loss, logitsPerClass * sizeof(float)));
    CUDA_CHECK(hipMalloc((void**) &d_count, logitsPerClass * sizeof(int)));
}

CUDACrossEntropyLoss::~CUDACrossEntropyLoss() {
    CUDA_CHECK(hipFree(d_loss));
    CUDA_CHECK(hipFree(d_count));
}

void CUDACrossEntropyLoss::forward(bool training) {
    timer_start(TMR_LOSS_FW);

    if (training) logits->zero_grad();
    
    int logitsPerClass = logits->size / num_classes;

    CUDA_CHECK(hipMemset(d_loss, 0, logitsPerClass * sizeof(float)));
    CUDA_CHECK(hipMemset(d_count, 0, logitsPerClass * sizeof(int)));

    dim3 block(logitsPerClass, 1, 1);
    dim3 thread_in_block(1, 1, 1);
    hipLaunchKernelGGL(cuda_CrossEntropy_forward_A_kernel,block, thread_in_block,0,0,logits->data, logits->grad, training, num_classes, truth, d_count, d_loss, logits->size);
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<int> count_ptr = thrust::device_pointer_cast(d_count);
    int count = thrust::reduce(count_ptr, count_ptr + logitsPerClass, (int)0, thrust::plus<int>());
    thrust::device_ptr<float> loss_ptr = thrust::device_pointer_cast(d_loss);
    *loss = thrust::reduce(loss_ptr, loss_ptr + logitsPerClass, (float)0.0, thrust::plus<float>());
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    *loss /= count;
    dim3 block2(logits->size, 1, 1);
    dim3 thread_in_block2(1, 1, 1);
    if (training) {
        hipLaunchKernelGGL(cuda_CrossEntropy_forward_B_kernel,block2, thread_in_block2,0,0,logits->grad, logits->size, count);
        CUDA_CHECK(hipGetLastError());
        // CUDA_CHECK(cudaDeviceSynchronize());
    }

    timer_stop(TMR_LOSS_FW);
}

void CUDACrossEntropyLoss::backward() {
}

CUDAReLU::CUDAReLU(CUDAVariable *in) :
    in(in) {
    CUDA_CHECK(hipMalloc((void**) &mask, in->size * sizeof(bool)));
}

CUDAReLU::~CUDAReLU() {
    CUDA_CHECK(hipFree(mask));
}

void CUDAReLU::forward(bool training) {
    timer_start(TMR_RELU_FW);

    dim3 block((in->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    hipLaunchKernelGGL(cuda_ReLU_forward_kernel,block, thread_in_block,0,0,in->data, mask, in->size, training);
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_RELU_FW);
}

void CUDAReLU::backward() {
    timer_start(TMR_RELU_BW);

    dim3 block((in->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    hipLaunchKernelGGL(cuda_ReLU_backward_kernel,block, thread_in_block,0,0,in->grad, mask, in->size);
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_RELU_BW);
}

CUDADropout::CUDADropout(CUDAVariable *in, float p) :
    in(in), p(p) {
    if (in->requires_grad) {
        CUDA_CHECK(hipMalloc((void**) &mask, in->size * sizeof(int)));
    }
    else
        mask = nullptr;
}

CUDADropout::~CUDADropout() {
    if (mask != nullptr) CUDA_CHECK(hipFree(mask));
}

void CUDADropout::forward(bool training) {
    if (!training) return;
    timer_start(TMR_DROPOUT_FW);

    float scale = 1 / (1 - p);
    dim3 block((in->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    hipLaunchKernelGGL(cuda_Dropout_forward_kernel,block, thread_in_block,0,0,in->data, mask, devStates, in->size, p, scale, (mask != nullptr));
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_DROPOUT_FW);
}

void CUDADropout::backward() {
    if (mask == nullptr) return;
    timer_start(TMR_DROPOUT_BW);
    
    float scale = 1 / (1 - p);
    dim3 block((in->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
    dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
    hipLaunchKernelGGL(cuda_Dropout_backward_kernel,block, thread_in_block,0,0,in->grad, mask, in->size, scale);
    CUDA_CHECK(hipGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    timer_stop(TMR_DROPOUT_BW);
}

CUDAAdamVariable::CUDAAdamVariable(CUDAVariable *var, bool decay) :
    data(var->data), grad(var->grad), size(var->size), decay(decay) {
    CUDA_CHECK(hipMalloc((void**) &m, size * sizeof(float)));
    CUDA_CHECK(hipMalloc((void**) &v, size * sizeof(float)));
}

CUDAAdamVariable::~CUDAAdamVariable() {
    CUDA_CHECK(hipFree(m));
    CUDA_CHECK(hipFree(v));
}

CUDAAdam::CUDAAdam(vector<pair<CUDAVariable*, bool>> vars, AdamParams params) :
    step_count(0), params(params){
    for (auto v : vars) {
        CUDAAdamVariable *adam_var = new CUDAAdamVariable(v.first, v.second);
        this->vars.push_back(adam_var);
    }
}

CUDAAdam::~CUDAAdam() {
    for (auto &var : vars)
        delete var;
}

void CUDAAdam::step() {
    step_count++;
    float step_size = params.lr * sqrtf(1 - powf(params.beta2, step_count)) / (1 - powf(params.beta1, step_count));
    for (auto &var : vars) {
        dim3 block((var->size-1)/MAX_THREAD_PER_BLOCK + 1, 1, 1);
        dim3 thread_in_block(MAX_THREAD_PER_BLOCK, 1, 1);
        hipLaunchKernelGGL(cuda_Adam_step_kernel,block, thread_in_block,0,0,var->grad, var->data, var->m, var->v, var->decay, params.weight_decay, params.beta1, params.beta2, params.eps, step_size, var->size);
        CUDA_CHECK(hipGetLastError());
        // CUDA_CHECK(cudaDeviceSynchronize());
    }
}
