#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("matmul_cuda(Tensor A, Tensor B) -> Tensor");
  ops.impl("matmul_cuda", torch::kCUDA, &matmul_cuda);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
