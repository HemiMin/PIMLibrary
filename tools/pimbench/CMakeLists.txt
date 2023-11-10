aux_source_directory(. pimbench_source)
aux_source_directory(source pimbench_source)

file(GLOB GCN_SEQ gcn/seq/*.cpp)
file(GLOB GCN_HIP gcn/hip/*.cpp)
file(GLOB GCN_COMMON gcn/common/*.cpp)
set(GCN_SRC ${GCN_SEQ} ${GCN_HIP} ${GCN_COMMON} )

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

include_directories(./include)
include_directories(./gcn/seq)
include_directories(./gcn/hip)
include_directories(./gcn/common)

add_compile_options(-Iinclude/gcn/common -Iinclude/gcn/seq -Iinclude/gcn/hip) 

add_executable(pimbench ${pimbench_source} ${GCN_SRC})
target_link_libraries(pimbench PimRuntime glog gflags Boost::program_options rocblas)
