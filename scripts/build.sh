#!/bin/bash
# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! getopt --test > /dev/null 
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=dfo:v
LONGOPTS=debug,output:,verbose

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

d=n v=n proj_cmake_dir=-
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -d|--debug)
            d=y
            shift
            ;;
        -v|--verbose)
            v=y
            shift
            ;;
        -o|--output)
            proj_cmake_dir="$(pwd)/$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

# handle non-option arguments
if [[ $# -ne 1 ]]; then
    echo "$0: A single input file is required."
    exit 4
fi

echo "verbose: $v, debug: $d, option: $1, out: $proj_cmake_dir"

cmake_build_options=""

if [ $d = "y" ]; then
    cmake_build_options="${cmake_build_options} -DCMAKE_BUILD_TYPE=Debug"
    echo "${cmake_build_options}"
else
    cmake_build_options="${cmake_build_options} -DCMAKE_BUILD_TYPE=Release"
fi

build="${proj_cmake_dir}/build"

if [ -d "$build" ]; then
    echo "build directory path ${build}"
else
    mkdir build
fi

cmake_fn()
{
    cd ${build}
    cmake ${cmake_build_options} ${proj_cmake_dir}
}

make_fn()
{
    cd ${build}
    make -j8
}

make_install_fn()
{
    cd ${build}
    make -j8
    sudo make install
}

uninstall_fn()
{
    sudo rm -f /opt/rocm/lib/libFimRuntime.so
    sudo rm -f /opt/rocm/include/fim_runtime_api.h
    sudo rm -f /opt/rocm/include/fim_data_types.h
    sudo rm -f /opt/rocm/lib/libdramsim2.so
    sudo rm -rf /opt/rocm/include/dramsim2
    sudo rm -rf /opt/rocm/lib/tf_fim_ops
}

if [ $1 = "all" ]; then
    uninstall_fn
    cmake_fn
    make_fn
    make_install_fn
fi

if [ $1 = "cmake" ]; then
    cmake_fn
fi

if [ $1 = "make" ]; then
    make_fn
fi

if [ $1 = "install" ]; then
    make_fn
    make_install_fn
fi

if [ $1 = "uninstall" ]; then
    uninstall_fn
fi