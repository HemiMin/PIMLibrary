import torch

torch.ops.load_library("/opt/rocm/lib/libpy_fim_eltwise.so")
py_fim_eltwise = torch.ops.custom_ops.py_fim_eltwise