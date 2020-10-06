import unittest
import numpy as np
import torch
import py_fim_ops


class PyFimMulTestConstant(unittest.TestCase):
    def test_vector_vector(self):
        input0 = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float16)
        input1 = torch.tensor([50., 5., 5., 5., 0.], dtype=torch.float16)
        mul = torch.tensor([1], dtype=torch.int32)

        result = py_fim_ops.py_fim_eltwise(input0, input1, mul)
        true_result = torch.tensor([50., 10., 15., 20., 0],
                                   dtype=torch.float16)
        assert (result.numpy() == true_result.numpy()).all()

    def test_scalar_vector(self):
        input0 = torch.tensor([20], dtype=torch.float16)
        input1 = torch.tensor([[1., 2., 3., 4., 0.], [6., 7., 8., 9., 1.]],
                              dtype=torch.float16)
        mul = torch.tensor([1], dtype=torch.int32)

        result = py_fim_ops.py_fim_eltwise(input0, input1, mul)
        true_result = torch.tensor(
            [[20., 40., 60., 80., 0.], [120., 140., 160., 180., 20]],
            dtype=torch.float16)
        assert (result.numpy() == true_result.numpy()).all()

    def test_vector_scalar(self):
        input0 = torch.tensor([[1., 2., 3., 4., 0.], [6., 7., 8., 9., 1.]],
                              dtype=torch.float16)
        input1 = torch.tensor([20], dtype=torch.float16)
        mul = torch.tensor([1], dtype=torch.int32)

        result = py_fim_ops.py_fim_eltwise(input0, input1, mul)
        true_result = torch.tensor(
            [[20., 40., 60., 80., 0.], [120., 140., 160., 180., 20]],
            dtype=torch.float16)

        assert (result.numpy() == true_result.numpy()).all()

    def test_scalar_scalar(self):
        input0 = torch.tensor([10], dtype=torch.float16)
        input1 = torch.tensor([100], dtype=torch.float16)
        mul = torch.tensor([1], dtype=torch.int32)

        result = py_fim_ops.py_fim_eltwise(input0, input1, mul)
        true_result = torch.tensor([1000], dtype=torch.float16)
        assert (result.numpy() == true_result.numpy()).all()

    def test_2Dscalar_vector(self):
        input0 = torch.tensor([[3]], dtype=torch.float16)
        input1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]],
                              dtype=torch.float16)
        mul = torch.tensor([1], dtype=torch.int32)

        result = py_fim_ops.py_fim_eltwise(input0, input1, mul)
        true_result = torch.tensor([[3, 6, 9, 12], [15, 18, 21, 24]],
                                   dtype=torch.float16)
        assert (result.numpy() == true_result.numpy()).all()


class PyFimMulTestRandom(unittest.TestCase):
    def test_4dim_4dim(self):
        batch_size = [1, 4, 8]
        channel = [1]
        width = [128, 256, 384]
        height = [768, 1024]
        failed_cases = []
        success = True

        for b in batch_size:
            for c in channel:
                for w in width:
                    for h in height:
                        min_val = -500
                        max_val = 500
                        input0 = np.random.uniform(min_val,
                                                   max_val,
                                                   size=[b, c, w,
                                                         h]).astype(np.float16)
                        input1 = np.random.uniform(min_val,
                                                   max_val,
                                                   size=[b, c, w,
                                                         h]).astype(np.float16)
                        mul = torch.tensor([1], dtype=torch.int32)

                        result = py_fim_ops.py_fim_eltwise(
                            torch.from_numpy(input0), torch.from_numpy(input1),
                            mul)
                        true_result = np.multiply(input0, input1)
                        try:
                            assert (result.numpy() == true_result).all()
                        except Exception as ex:
                            failed_cases.append([b, c, w, h])
                            success = False
        if not success:
            print("Test cases failed!: " + str(failed_cases))
        assert (success == True)

    def test_4dim_scalar(self):
        batch_size = [1, 8]
        channel = [1]
        width = [128]
        height = [768, 1024]
        failed_cases = []
        success = True

        for b in batch_size:
            for c in channel:
                for w in width:
                    for h in height:
                        min_val = -500
                        max_val = 500
                        input0 = np.random.uniform(min_val,
                                                   max_val,
                                                   size=[b, c, w,
                                                         h]).astype(np.float16)
                        input1 = np.array([100]).astype(np.float16)
                        mul = torch.tensor([1], dtype=torch.int32)

                        result = py_fim_ops.py_fim_eltwise(
                            torch.from_numpy(input0), torch.from_numpy(input1),
                            mul)
                        true_result = np.multiply(input0, input1)
                        try:
                            assert (result.numpy() == true_result).all()
                        except Exception as ex:
                            failed_cases.append([b, c, w, h])
                            success = False
        if not success:
            print("Test cases failed!: " + str(failed_cases))
        assert (success == True)


class PyFimMulTestFile(unittest.TestCase):
    def test1(self):
        input0 = np.fromfile("test_vectors/load/elt_mul/input0_256KB.dat",
                             dtype=np.float16)
        input1 = np.fromfile("test_vectors/load/elt_mul/input1_256KB.dat",
                             dtype=np.float16)

        mul = torch.tensor([1], dtype=torch.int32)
        result = py_fim_ops.py_fim_eltwise(torch.from_numpy(input0),
                                           torch.from_numpy(input1), mul)
        golden = np.fromfile("test_vectors/load/elt_mul/output_256KB.dat",
                             dtype=np.float16)
        assert (result.numpy() == golden).all()

    def test2(self):
        input0 = np.fromfile("test_vectors/load/elt_mul/input0_512KB.dat",
                             dtype=np.float16)
        input1 = np.fromfile("test_vectors/load/elt_mul/input1_512KB.dat",
                             dtype=np.float16)

        mul = torch.tensor([1], dtype=torch.int32)
        result = py_fim_ops.py_fim_eltwise(torch.from_numpy(input0),
                                           torch.from_numpy(input1), mul)
        golden = np.fromfile("test_vectors/load/elt_mul/output_512KB.dat",
                             dtype=np.float16)
        assert (result.numpy() == golden).all()

    def test_scalar_vector(self):
        input0 = np.fromfile("test_vectors/load/sv_mul/scalar_2B.dat",
                             dtype=np.float16)
        input1 = np.fromfile("test_vectors/load/sv_mul/vector_256KB.dat",
                             dtype=np.float16)

        mul = torch.tensor([1], dtype=torch.int32)
        result = py_fim_ops.py_fim_eltwise(torch.from_numpy(input0),
                                           torch.from_numpy(input1), mul)
        golden = np.fromfile("test_vectors/load/sv_mul/output_256KB.dat",
                             dtype=np.float16)
        assert (result.numpy() == golden).all()


if __name__ == '__main__':

    py_fim_ops.py_fim_init()
    unittest.main()
    py_fim_ops.py_fim_deinit()