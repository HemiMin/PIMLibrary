#!/bin/bash
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
