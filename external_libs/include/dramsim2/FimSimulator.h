#ifndef __FIM_SIMULATOR_HPP__
#define __FIM_SIMULATOR_HPP__

#include "FIMCtrl.h"

class FimSimulator
{
   public:
    FimSimulator();
    ~FimSimulator();
    void initialize(const string& device_ini_file_name, const string& system_ini_file_name, size_t megs_of_memory,
                    size_t num_fim_chan, size_t num_fim_rank);
    void preload_data(void* data, size_t data_size);
    void preload_data_with_addr(uint64_t addr, void* data, size_t data_size);
    void execute_kernel(void* trace_data, size_t num_trace);
    void execute_kernel_bn(void* trace_data, size_t num_trace, int num_batch, int num_ch, int num_width);
    void alloc_burst(size_t preload_size, size_t output_size);
    void get_uint16_result(uint16_t* output_data, size_t num_data);
    void get_uint16_result_gemv(uint16_t* output_data, size_t num_data);
    void read_result(uint64_t addr, size_t data_size);
    void run();
    void compare_result_arr(uint16_t* test_output, size_t num_data, NumpyBurstType* output_npbst);

    // function for test
    void vector_to_arr(vector<MemTraceData>& vec_trace_data, MemTraceData* trace_data);
    void set_data_for_eltwise(NumpyBurstType* input0, NumpyBurstType* input1, uint16_t* test_input);
    void set_data_for_bn(NumpyBurstType* input0, uint16_t* test_input);
    void compare_result(size_t num_data, NumpyBurstType* output_npbst);
    void read_memory_trace(const string& file_name, vector<MemTraceData>& vec_trace_data);
    void create_tv_for_gemv_test(NumpyBurstType* weight_npbst, uint16_t* test_weight);

   private:
    void convert_arr_to_burst(void* data, size_t data_size, BurstType* bst);
    void push_trace(vector<TraceDataBst>* trace_bst);
    void push_trace_bn(vector<TraceDataBst>* trace_bst, int num_batch, int num_ch, int num_width);

    void convert_to_burst_trace(void* trace_data, vector<TraceDataBst>* trace_bst, size_t num_trace);

   private:
    shared_ptr<FIMController> fim_controller_;  // for test
    shared_ptr<MultiChannelMemorySystem> mem_;

    BurstType* preload_burst_;
    BurstType* output_burst_;
    int bst_size_;
    size_t cycle_;
};

#endif
