
#include "common_perf.h"
#include "pim_data_types.h"

class PimGCNTest
{
   public:
    PimGCNTest();
    ~PimGCNTest();
    void prepare(float variation = 0.01f);
    void execute_op(bool block = true);
    void finalize();
    int validate(float epsilon = 1e-2);

   private:
};

class PimGCNTestFixture : public PerformanceAnalyser
{
   public:
    PimGCNTestFixture();

   protected:
    bool has_bias;

    int ExecuteTest();
};

