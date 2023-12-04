#!/bin/bash

if grep -rni "rocrand/rocrand.h" /opt/rocm-4.0.0/include/hiprand/hiprand_hcc.h; then
  echo "RocRand Lib is already modified"
else
  sudo ./modifyRandLib.sh
fi

./scripts/build.sh cmake -o . -t amd
./scripts/build.sh install -o . -t amd

i_h=(1 4 10 20 40)
i_w=(256 512 1024 1536 2048 4096)
o_w=(4096)

mkdir GemmRes
for ih in "${i_h[@]}"; do
for iw in "${i_w[@]}"; do
for ow in "${o_w[@]}"; do
  sudo ./build/tools/pimbench/pimbench -o gemm -n 1 -c 1 --i_h ${ih} --i_w ${iw} --o_h ${ih} --o_w ${ow} --order i_x_w -i 5 &> GemmRes/gemm_${ih}_${iw}_${ih}_${ow}
done
done
done

python3 variousGemm_res.py

sudo ./build/tools/pimbench/pimbench -o gcn -i 5 &> gcn_res
python3 gcn_res.py

mkdir PimTypeGemmRes
./gemm_devicetype_to_pim.sh
./scripts/build.sh install -o . -t amd
for ih in "${i_h[@]}"; do
for iw in "${i_w[@]}"; do
for ow in "${o_w[@]}"; do
  sudo ./build/tools/pimbench/pimbench -o gemm -n 1 -c 1 --i_h ${ih} --i_w ${iw} --o_h ${ih} --o_w ${ow} --order i_x_w -i 5 &> PimTypeGemmRes/gemm_${ih}_${iw}_${ih}_${ow}
done
done
done
python3 variousPimTypeGemm_res.py
./gemm_pimtype_to_device.sh
./scripts/build.sh install -o . -t amd
