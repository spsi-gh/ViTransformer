#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu(float x) {
    const float pi = 3.14159265359f;
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / pi) * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_forward_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = gelu(input[idx]);
}

torch::Tensor gelu_forward(torch::Tensor input) {
    input = input.contiguous();
    auto output = torch::empty_like(input);
    size_t size = input.numel();

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    gelu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_forward", &gelu_forward, "GELU forward (CUDA)");
}
