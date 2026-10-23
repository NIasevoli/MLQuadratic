#include <torch/torch.h>
#include <iostream>

int main() 
{
  torch::Tensor altezza = torch::tensor(
      {1.23f, 44.15f, 176.58f, 595.06f, 828.61f, 962.34f, 1103.63f, 2163.0f},
      torch::kFloat32).unsqueeze(1);
  torch::Tensor tempo = torch::tensor(
      {0.5f, 3.0f, 6.0f, 11.0f, 13.0f, 14.0f, 15.0f, 21.0f},
      torch::kFloat32).unsqueeze(1);

  torch::manual_seed(42);
  torch::nn::Sequential model(
      torch::nn::Linear(1, 10),
      torch::nn::ReLU(),
      torch::nn::Linear(10, 10),
      torch::nn::ReLU(),
      torch::nn::Linear(10, 1));
  torch::nn::MSELoss loss_func;
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.01));

  torch::Tensor tempo_pred;
  for (int epoch = 0; epoch < 3000; ++epoch) {
    optimizer.zero_grad();
    tempo_pred = model->forward(altezza);
    torch::Tensor loss = loss_func(tempo_pred, tempo);
    loss.backward();
    optimizer.step();

    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << ", loss: " << loss.item<float>() << std::endl;
    }
  }

  // Equivalente del plot: stampa dati reali e predizioni finali
  std::cout << "altezza | tempo | tempo_pred\n";
  auto pred = tempo_pred.detach();
  for (int i = 0; i < altezza.size(0); ++i) {
    std::cout << altezza[i].item<float>() << " | "
              << tempo[i].item<float>() << " | "
              << pred[i].item<float>() << "\n";
  }
  return 0;
}