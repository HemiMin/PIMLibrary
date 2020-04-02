#include "manager/FimMemoryManager.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "utility/fim_util.h"

namespace fim
{
namespace runtime
{
namespace manager
{
FimMemoryManager::FimMemoryManager(FimDevice* fim_device, FimRuntimeType rt_type, FimPrecision precision)
    : fim_device_(fim_device), rt_type_(rt_type), precision_(precision)
{
    DLOG(INFO) << "called";
    get_fim_block_info(&fbi_);
}

FimMemoryManager::~FimMemoryManager(void) { DLOG(INFO) << "called"; }
int FimMemoryManager::initialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

int FimMemoryManager::deinitialize(void)
{
    DLOG(INFO) << "called";
    int ret = 0;

    return ret;
}

int FimMemoryManager::alloc_memory(void** ptr, size_t size, FimMemType mem_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (mem_type == MEM_TYPE_DEVICE) {
        if (hipMalloc((void**)ptr, size) != hipSuccess) {
            return -1;
        }
    } else if (mem_type == MEM_TYPE_HOST) {
        *ptr = (void*)malloc(size);
    } else if (mem_type == MEM_TYPE_FIM) {
        *ptr = fragment_allocator_.alloc(size);
    }

    return ret;
}

int FimMemoryManager::alloc_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (fim_bo->mem_type == MEM_TYPE_DEVICE) {
        if (hipMalloc((void**)&fim_bo->data, fim_bo->size) != hipSuccess) {
            return -1;
        }
    } else if (fim_bo->mem_type == MEM_TYPE_HOST) {
        fim_bo->data = (void*)malloc(fim_bo->size);
    } else if (fim_bo->mem_type == MEM_TYPE_FIM) {
        fim_bo->data = fragment_allocator_.alloc(fim_bo->size);
    }

    return ret;
}

int FimMemoryManager::free_memory(void* ptr, FimMemType mem_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (mem_type == MEM_TYPE_DEVICE) {
        if (hipFree(ptr) != hipSuccess) {
            return -1;
        }
    } else if (mem_type == MEM_TYPE_HOST) {
        free(ptr);
    } else if (mem_type == MEM_TYPE_FIM) {
        return fragment_allocator_.free(ptr);
    }

    return ret;
}

int FimMemoryManager::free_memory(FimBo* fim_bo)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (fim_bo->mem_type == MEM_TYPE_DEVICE) {
        if (hipFree(fim_bo->data) != hipSuccess) {
            return -1;
        }
    } else if (fim_bo->mem_type == MEM_TYPE_HOST) {
        free(fim_bo->data);
    } else if (fim_bo->mem_type == MEM_TYPE_FIM) {
        if (fragment_allocator_.free(fim_bo->data)) return 0;
        return -1;
    }

    return ret;
}

int FimMemoryManager::copy_memory(void* dst, void* src, size_t size, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (cpy_type == HOST_TO_FIM || cpy_type == HOST_TO_DEVICE) {
        if (hipMemcpy(dst, src, size, hipMemcpyHostToDevice) != hipSuccess) {
            return -1;
        }
    } else if (cpy_type == FIM_TO_HOST || cpy_type == DEVICE_TO_HOST) {
        if (hipMemcpy(dst, src, size, hipMemcpyDeviceToHost) != hipSuccess) {
            return -1;
        }
    } else if (cpy_type == DEVICE_TO_FIM || cpy_type == FIM_TO_DEVICE) {
        if (hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice) != hipSuccess) {
            return -1;
        }
    } else if (cpy_type == HOST_TO_HOST) {
        if (hipMemcpy(dst, src, size, hipMemcpyHostToHost) != hipSuccess) {
            return -1;
        }
    }

    return ret;
}

int FimMemoryManager::copy_memory(FimBo* dst, FimBo* src, FimMemCpyType cpy_type)
{
    DLOG(INFO) << "called";
    int ret = 0;
    size_t size = dst->size;

    if (cpy_type == HOST_TO_FIM || cpy_type == HOST_TO_DEVICE) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyHostToDevice) != hipSuccess) {
            return -1;
        }
    } else if (cpy_type == FIM_TO_HOST || cpy_type == DEVICE_TO_HOST) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyDeviceToHost) != hipSuccess) {
            return -1;
        }
    } else if (cpy_type == DEVICE_TO_FIM || cpy_type == FIM_TO_DEVICE) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyDeviceToDevice) != hipSuccess) {
            return -1;
        }
    } else if (cpy_type == HOST_TO_HOST) {
        if (hipMemcpy(dst->data, src->data, size, hipMemcpyHostToHost) != hipSuccess) {
            return -1;
        }
    }

    return ret;
}

int FimMemoryManager::convert_data_layout(void* dst, void* src, size_t size, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (op_type == OP_GEMV) {
    } else {
        DLOG(ERROR) << "not yet implemented";
    }

    return ret;
}

int FimMemoryManager::convert_data_layout(FimBo* dst, FimBo* src, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;
    size_t size = dst->size;

    if (op_type == OP_GEMV) {
        ret = convert_data_layout_for_gemv_weight(dst, src);
    } else {
        DLOG(ERROR) << "not yet implemented";
    }

    return ret;
}

int FimMemoryManager::convert_data_layout(FimBo* dst, FimBo* src0, FimBo* src1, FimOpType op_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    if (op_type == OP_ELT_ADD || op_type == OP_ELT_MUL) {
        ret = convert_data_layout_for_elt_add(dst, src0, FimBankType::EVEN_BANK);
        ret = convert_data_layout_for_elt_add(dst, src1, FimBankType::ODD_BANK);
    } else {
        DLOG(ERROR) << "not yet implemented";
    }

    return ret;
}

int FimMemoryManager::convert_data_layout_for_gemv_weight(FimBo* dst, FimBo* src)
{
    DLOG(INFO) << "called";
    int ret = 0;

    int num_grf_A = fbi_.num_grf;
    int num_grf_B = fbi_.num_grf;
    int num_fim_blocks = fbi_.num_fim_blocks;
    int num_fim_chan = fbi_.num_fim_chan;
    int num_fim_rank = fbi_.num_fim_rank;
    int num_banks = fbi_.num_banks;
    int num_bank_groups = fbi_.num_bank_groups;
    int trans_size = fbi_.trans_size;

    int in_tile_size = num_grf_A;
    int out_tile_size = num_grf_B * num_fim_blocks * num_fim_chan * num_fim_rank;
    char* dst_data = (char*)dst->data;
    char* src_data = (char*)src->data;

    int cidx = 0;
    int rank = 0;
    int bg = 0;
    int bank = 0;
    uint32_t col = 0;
    uint32_t row = 0;
    uint64_t addr;
    uint32_t even_s_row = 0;  // starting_row;
    uint32_t even_s_col = 0;  // starting_col;
    uint32_t odd_s_row = 0;   // starting_row;
    uint32_t odd_s_col = 0;   // starting_col;

    int type_size = (src->precision == FIM_FP16) ? 2 : 1;
    int out_cnt = src->bshape.h;
    int in_cnt = src->bshape.w * type_size / trans_size;

    for (int y = 0; y < out_cnt; y += out_tile_size) {
        for (int x = 0; x < in_cnt; x += in_tile_size) {
            if ((x / in_tile_size) % 2 == 0) {
                for (int tiled_y = 0; tiled_y < out_tile_size; tiled_y += num_grf_B) {
                    col = even_s_col;
                    row = even_s_row;

                    for (int grfb_idx = 0; grfb_idx < num_grf_B; grfb_idx++) {
                        for (int grfa_idx = 0; grfa_idx < num_grf_A; grfa_idx++) {
                            addr = addr_gen_safe(cidx, rank, bg, bank, row, col);
                            int d_idx = (y + tiled_y + grfa_idx) * in_cnt + x + grfb_idx;
                            memcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size);
                            col++;
                        }
                    }

                    bank += (num_banks / num_fim_blocks);

                    if (bank >= (num_banks / num_bank_groups)) {
                        bg++;
                        bank = 0;
                    }

                    if (bg >= num_bank_groups) {
                        bg = 0;
                        rank++;
                    }

                    if (rank >= num_fim_rank) {
                        rank = 0;
                        cidx++;
                    }

                    if (cidx >= num_fim_chan) {
                        cidx = 0;
                        even_s_row = row;
                        even_s_col = col;
                    }
                }
            } else if ((x / in_tile_size) % 2 == 1) {
                for (int tiled_y = 0; tiled_y < out_tile_size; tiled_y += num_grf_B) {
                    col = odd_s_col;
                    row = odd_s_row;

                    for (int grfb_idx = 0; grfb_idx < num_grf_B; grfb_idx++) {
                        for (int grfa_idx = 0; grfa_idx < num_grf_A; grfa_idx++) {
                            addr = addr_gen_safe(cidx, rank, bg, bank + 1, row, col);
                            int d_idx = (y + tiled_y + grfa_idx) * in_cnt + x + grfb_idx;
                            memcpy(dst_data + addr, src_data + d_idx * trans_size, trans_size);
                            col++;
                        }
                    }

                    bank += (num_banks / num_fim_blocks);

                    if (bank >= (num_banks / num_bank_groups)) {
                        bg++;
                        bank = 0;
                    }

                    if (bg >= num_bank_groups) {
                        bg = 0;
                        rank++;
                    }

                    if (rank >= num_fim_rank) {
                        rank = 0;
                        cidx++;
                    }

                    if (cidx >= num_fim_chan) {
                        cidx = 0;
                        odd_s_row = row;
                        odd_s_col = col;
                    }
                }
            }
        }
    }

    return ret;
}

int FimMemoryManager::convert_data_layout_for_elt_add(FimBo* dst, FimBo* src, FimBankType fim_bank_type)
{
    DLOG(INFO) << "called";
    int ret = 0;

    uint32_t cidx = 0;
    uint32_t rank = 0;
    uint32_t bg = 0;
    uint32_t bank = 0;
    uint32_t s_row = 0;
    uint32_t s_col = 0;
    uint32_t trans_size = fbi_.trans_size;
    uint32_t type_size = (precision_ == FIM_FP16) ? sizeof(half) : sizeof(char);
    uint64_t addr_op = 0x0;
    uint32_t dim_operand = dst->size / trans_size / type_size;
    char* dst_data = (char*)dst->data;
    char* src_data = (char*)src->data;
    int num_grf = fbi_.num_grf;
    int num_banks = fbi_.num_banks;
    int num_fim_blocks = fbi_.num_fim_blocks;
    int num_bank_groups = fbi_.num_bank_groups;
    int num_fim_rank = fbi_.num_fim_rank;
    int num_fim_chan = fbi_.num_fim_chan;

    for (int x = 0; x < dim_operand; x += num_grf) {
        uint32_t row = 0;
        uint32_t col = 0;

        for (int grf_idx = 0; grf_idx < num_grf; grf_idx++) {
            addr_op = addr_gen_safe(cidx, rank, bg, bank + (int)fim_bank_type, row, col);
            memcpy(dst_data + addr_op, src_data + (x + grf_idx) * trans_size, trans_size);
            col++;
        }

        bank += (num_banks / num_fim_blocks);

        if (bank >= (num_banks / num_bank_groups)) {
            bg++;
            bank = 0;
        }

        if (bg >= num_bank_groups) {
            bg = 0;
            rank++;
        }

        if (rank >= num_fim_rank) {
            rank = 0;
            cidx++;
        }

        if (cidx >= num_fim_chan) {
            cidx = 0;
            s_row = row;
            s_col = col;
        }
    }

    return ret;
}

void* FimMemoryManager::FimBlockAllocator::alloc(size_t request_size, size_t& allocated_size) const
{
    assert(request_size <= block_size() && "BlockAllocator alloc request exceeds block size.");
    void* ret = nullptr;
    size_t bsize = block_size();

    /* todo:implement fimalloc function */
    ret = (void*)malloc(bsize);

    assert(ret != nullptr && "Region returned nullptr on success.");

    allocated_size = block_size();
    return ret;
}

void FimMemoryManager::FimBlockAllocator::free(void* ptr, size_t length) const
{
    if (ptr == NULL || length == 0) {
        return;
    }

    /* todo:implement fimfree function */
    std::free(ptr);
}

} /* namespace manager */
} /* namespace runtime */
} /* namespace fim */
