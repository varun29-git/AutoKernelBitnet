#pragma once
#include <torch/torch.h>

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
