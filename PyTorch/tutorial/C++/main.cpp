#include <torch/torch.h>
#include <iostream>

/**
 * Defining a module and registering parameters
 */
struct NetSimple : torch::nn::Module {
    torch::Tensor W, b;

    NetSimple(int64_t N, int64_t M) {
        W = register_parameter("W", torch::randn({N, M}));
        b = register_parameter("b", torch::randn(M));
    }

    torch::Tensor forward(torch::Tensor input) {
        return torch::addmm(b, input, W);
    }
};

/**
 * Registering submodules and traversing the module Hierarchy
 */
struct NetImpl : torch::nn::Module {
    torch::nn::Linear linear;
    torch::Tensor another_bias;

    // NOTE: Constructing linear in the initializer list allows us to recursively access the module tree's parameters
    NetImpl(int64_t N, int64_t M)
        : linear(register_module("linear", torch::nn::Linear(N, M)))
    {
        another_bias = register_parameter("b", torch::randn(M));
    }

    torch::Tensor forward(torch::Tensor input) {
        return linear(input) + another_bias;
    }
};

/**
 * Passing the nn modules around in C++ is difficult because of reference and pointer logic.
 * This can be mitigated by using shared_ptr and PyTorch gives a clean way to implement this
 */
 TORCH_MODULE(Net);  // This macro defines a module called "Linear" from "LinearImpl". This generated class is effectively a wrapper over an std::shared_ptr<LinearImpl>

int main() {
    Net net(4, 5);

    // Use 'net.parameters' to get only the values
    for (const auto& pair :  net.named_parameters()) {
        std::cout << pair.key() << ": " << std::endl << pair.value() << std::endl;
    }

    // Feed forward
    std::cout << net.forward(torch::ones({2, 4})) << std::endl;

    return 0;
}
