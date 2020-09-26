#include <cmath>
#include <torch/torch.h>
#include <iostream>

const int64_t kNoiseSize = 100;                   // The size of the noise vector fed to the generator
const int64_t kBatchSize = 64;                    // The batch size for training
const int64_t kNumberOfEpochs = 30;               // The number of epochs to train
const int64_t kCheckpointEvery = 200;             // After how many batches to create a new checkpoint periodically
const int64_t kNumberOfSamplesPerCheckpoint =10;  // How many images to sample at every checkpoint
const int64_t kLogInterval = 10;                  // After how many batches to log a new update with the loss value.
const bool kRestoreFromCheckpoint = false;        // Set to 'true' to restore models and optimizers from previously saved checkpoints
const char* kDataFolder = "./data";               // Where to find the MNIST dataset

struct DCGANGeneratorImpl : torch::nn::Module {
    torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4;
    torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;

    DCGANGeneratorImpl(int kNoiseSize)
        : conv1(torch::nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
          batch_norm1(256),
          conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
          batch_norm2(128),
          conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1).bias(false)),
          batch_norm3(64),
          conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1).bias(false))
    {
        // register_module() is needed if we want to use the parameteres() method later
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::tanh(conv4(x));

        return x;
    }
};
TORCH_MODULE(DCGANGenerator);

DCGANGenerator generator();

int main() {
    return 0;
}
