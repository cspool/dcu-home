import torch
import torch.nn.functional as F


def main() -> None:
    n, k, m = 512, 5120, 34816
    print("torch", torch.__version__)
    print("arch", torch.cuda.get_device_properties(0).gcnArchName)
    print("preferred_blas", torch.backends.cuda.preferred_blas_library())

    x = torch.ones((n, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
    output = F.linear(x, weight)
    torch.cuda.synchronize()
    print("shape", tuple(x.shape), tuple(weight.shape), tuple(output.shape))
    print("dtype", output.dtype, "sample", float(output[0, 0]))


if __name__ == "__main__":
    main()
