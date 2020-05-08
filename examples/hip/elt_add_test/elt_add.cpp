#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"

#define CHECK(cmd)                                                                                              \
    {                                                                                                           \
        hipError_t error = cmd;                                                                                 \
        if (error != hipSuccess) {                                                                              \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
    }

__device__ inline void R_CMD(uint8_t* addr)
{
    asm volatile("global_load_dwordx4 v[24:27], %0, off\n\t" ::"v"(addr) : "v24", "v25", "v26", "v27");
}

__device__ inline void W_CMD(uint8_t* addr)
{
    asm volatile("global_store_dwordx4 %0, v[24:27], off\n\t" ::"v"(addr) : "v24", "v25", "v26", "v27");
}

__device__ inline void W_CMD_R(uint8_t* addr, uint8_t* src)
{
    if (hipThreadIdx_x == 0) {
        asm volatile("global_load_dwordx4 v[20:23], %0, off\n\t" ::"v"(src) : "v20", "v21", "v22", "v23");
        asm volatile("global_store_dwordx4 %0, v[20:23], off\n\t" ::"v"(addr) : "v20", "v21", "v22", "v23");
    } else {
        asm volatile("global_load_dwordx4 v[24:27], %0, off\n\t" ::"v"(src) : "v24", "v25", "v26", "v27");
        asm volatile("global_store_dwordx4 %0, v[24:27], off\n\t" ::"v"(addr) : "v24", "v25", "v26", "v27");
    }
}

__device__ inline void B_CMD(int type)
{
    if (type == 0)
        __syncthreads();
    else
        __threadfence();
}

__device__ inline unsigned int mask_by_bit(unsigned int value, int start, int end)
{
    int length = start - end + 1;
    value = value >> end;
    return value & ((1 << length) - 1);
}

__device__ uint64_t addr_gen(unsigned int ch, unsigned int rank, unsigned int bg, unsigned int ba, unsigned int row,
                             unsigned int col)
{
    int num_row_bit_ = 14;
    int num_col_high_bit_ = 3;
    int num_bank_high_bit_ = 1;
    int num_bankgroup_bit_ = 2;
    int num_bank_low_bit_ = 1;
    int num_chan_bit_ = 6;
    int num_col_low_bit_ = 1;
    int num_offset_bit_ = 5;

    uint64_t addr = rank;

    addr <<= num_row_bit_;
    addr |= row;

    addr <<= num_col_high_bit_;
    addr |= mask_by_bit(col, 4, 2);  // HARDCODED. FIXME

    addr <<= num_bank_high_bit_;
    addr |= mask_by_bit(ba, 1, 1);

    addr <<= num_bankgroup_bit_;
    addr |= bg;

    addr <<= num_bank_low_bit_;
    addr |= mask_by_bit(ba, 0, 0);

    addr <<= num_chan_bit_ - 1;
    addr |= mask_by_bit(ch, num_chan_bit_ - 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(col, 1, 1);

    addr <<= 1;
    addr |= mask_by_bit(ch, 0, 0);

    addr <<= 1;
    addr |= mask_by_bit(col, 0, 0);

    addr <<= num_offset_bit_;

    return addr;
}

__global__ void compute_add(uint8_t* fim_data, uint8_t* fim_ctr, uint8_t* output, unsigned int input_size,
                            uint8_t* crf_binary, int crf_size, uint8_t* hab_to_fim, uint8_t* fim_to_hab)
{
    int num_blk = 8;
    int num_g = 8;
    int num_bg = 4;
    int num_ba = 16;
    int num_ch = hipGridDim_x;
    int num_rank = 1;
    int num_col = 32;
    int g_size = 32;
    int i_size = input_size / sizeof(uint16_t);
    unsigned int start_row = (((uint64_t)fim_data - (uint64_t)fim_ctr) >> 19) & 0x3FFF;

    int num_p = num_blk * num_ch * num_g * (g_size / sizeof(short));
    int num_t = i_size / num_p;

    uint64_t addr;
    uint64_t offset = (hipThreadIdx_x % 2) * 0x10;

    B_CMD(0);
    for (int tile_idx = 0; tile_idx < num_t; tile_idx++) {
        unsigned int loc = tile_idx * num_g + (hipThreadIdx_x / 2);
        unsigned int row = start_row + loc / num_col;
        unsigned int col = loc % num_col;

        addr = addr_gen(hipBlockIdx_x, 0, 0, 0, row, col);
        R_CMD(&fim_ctr[addr + offset]);
        B_CMD(0);

        R_CMD(&fim_ctr[addr + 0x2000 + offset]);
        B_CMD(0);

        unsigned int output_loc = loc + num_t * num_g;
        unsigned int output_row = start_row + output_loc / num_col;
        unsigned int output_col = output_loc % num_col;

        addr = addr_gen(hipBlockIdx_x, 0, 0, 1, output_row, output_col);
        W_CMD(&fim_ctr[addr + offset]);
        B_CMD(0);
    }
}

int main(int argc, char* argv[])
{
    uint64_t* fim_base;
    uint64_t *mode1_d, *mode2_d, *crf_bin_d;
    uint64_t *mode1_h, *mode2_h, *crf_bin_h;
    size_t N = 4;
    size_t Nbytes = N * sizeof(uint64_t);
    static int device = 0;

    unsigned int input_size = 64 * 1024 * sizeof(uint16_t);

    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));

    CHECK(hipMalloc(&fim_base, 2 * 1024 * 64 * sizeof(uint16_t)));

    crf_bin_h = (uint64_t*)malloc(Nbytes);
    CHECK(crf_bin_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    mode1_h = (uint64_t*)malloc(Nbytes);
    CHECK(mode1_h == 0 ? hipErrorOutOfMemory : hipSuccess);
    mode2_h = (uint64_t*)malloc(Nbytes);
    CHECK(mode2_h == 0 ? hipErrorOutOfMemory : hipSuccess);

    crf_bin_h[0] = 0x1b18800098800000;
    crf_bin_h[1] = 0xe000000400000007;
    crf_bin_h[2] = 0x00000000f0000000;
    crf_bin_h[3] = 0x0000000000000000;
    mode1_h[0] = 0x0000000000000001;
    mode1_h[1] = 0x0000000000000000;
    mode1_h[2] = 0x0000000000000000;
    mode1_h[3] = 0x0000000000000000;
    mode2_h[0] = 0x0000000000000000;
    mode2_h[1] = 0x0000000000000000;
    mode2_h[2] = 0x0000000000000000;
    mode2_h[3] = 0x0000000000000000;

    CHECK(hipMalloc(&crf_bin_d, Nbytes));
    CHECK(hipMalloc(&mode1_d, Nbytes));
    CHECK(hipMalloc(&mode2_d, Nbytes));

    CHECK(hipMemcpy(crf_bin_d, crf_bin_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode1_d, mode1_h, Nbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(mode2_d, mode2_h, Nbytes, hipMemcpyHostToDevice));

    const unsigned blocks = 64;
    const unsigned threadsPerBlock = 16;

    hipLaunchKernelGGL(compute_add, dim3(blocks), dim3(threadsPerBlock), 0, 0, (uint8_t*)fim_base, (uint8_t*)fim_base,
                       (uint8_t*)0, input_size, (uint8_t*)crf_bin_d, 1, (uint8_t*)mode1_d, (uint8_t*)mode2_d);

    hipDeviceSynchronize();

    free(mode1_h);
    free(mode2_h);
    free(crf_bin_h);

    hipFree(mode1_d);
    hipFree(mode2_d);
    hipFree(crf_bin_d);
    hipFree(fim_base);
}
