#include <torch/torch.h>
#include <iostream>

int main(int argc, char** argv) 
{
  int epocs = 3000;
  float learningRate = 0.01;

  if(argc == 3)
  {
    std::cout << "*** argv[1]: " << argv[1] << std::endl;
    epocs = atoi(argv[2]);
    std::cout << "*** argv[2]: " << argv[2] << std::endl;
    learningRate = atof(argv[1]);
  }
    
  std::cout << "**** Epocs:" << epocs << "; learningRate = " << learningRate << std::endl;

  torch::Tensor altezza = torch::tensor(
      {1.23f, 44.15f, 176.58f, 595.06f, 828.61f, 962.34f, 1103.63f, 2163.0f, 2825.28f, 4414.5f},
      torch::kFloat32).unsqueeze(1);
  torch::Tensor tempo = torch::tensor(
      {0.5f, 3.0f, 6.0f, 11.0f, 13.0f, 14.0f, 15.0f, 21.0f, 24.0f, 30.0f},
      torch::kFloat32).unsqueeze(1);
  
  // Modello alternativo

  // Normalizzazione
  const float tScale = 30.0f, hScale = 4414.5f;
  auto tn = tempo / tScale;
  auto hn = altezza / hScale;
  torch::manual_seed(42);

  torch::nn::Sequential model(
    torch::nn::Linear(1, 32),
    torch::nn::Tanh(),            // più adatta di ReLU a una curva liscia
    torch::nn::Linear(32, 32),
    torch::nn::Tanh(),
    torch::nn::Linear(32, 1));

  torch::nn::MSELoss loss_func;
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learningRate));
  torch::Tensor tempo_pred;
  torch::Tensor loss;

  int epoch;
  for (epoch = 0; epoch < epocs; ++epoch) {
    optimizer.zero_grad();
    tempo_pred = model->forward(hn);
    //torch::Tensor loss = loss_func(tempo_pred, tempo);
    loss = loss_func(tempo_pred, tn);
    loss.backward();
    optimizer.step();

    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << ", loss: " << loss.item<float>() << std::endl;
    }
  }
  std::cout << "Epoch " << epoch << ", loss: " << loss.item<float>() << std::endl;

  // Equivalente del plot: stampa dati reali e predizioni finali
  std::cout << "altezza | tempo | tempo_pred\n";
  auto pred = tempo_pred.detach();
  for (int i = 0; i < hn.size(0); ++i) {
    std::cout << hn[i].item<float>() * hScale << " | "
              << tn[i].item<float>() * tScale << " | "
              << pred[i].item<float>() * tScale << "\n";
  }


  return 0;
}
