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
  

  torch::manual_seed(42);
  torch::nn::Sequential model(
      torch::nn::Linear(1, 10),
      torch::nn::ReLU(),
      torch::nn::Linear(10, 10),
      torch::nn::ReLU(),
      torch::nn::Linear(10, 1));

  

  torch::nn::MSELoss loss_func;
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learningRate));
  //torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.01));

  torch::Tensor tempo_pred;
  //for (int epoch = 0; epoch < epocs; ++epoch) {
  int epoch;
  torch::Tensor loss;
  for (epoch = 0; epoch < epocs; ++epoch) {
    optimizer.zero_grad();
    tempo_pred = model->forward(altezza);
    //torch::Tensor loss = loss_func(tempo_pred, tempo);
    loss = loss_func(tempo_pred, tempo);
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
  for (int i = 0; i < altezza.size(0); ++i) {
    std::cout << altezza[i].item<float>() << " | "
              << tempo[i].item<float>() << " | "
              << pred[i].item<float>() << "\n";
  }

  // Modello alternativo

  // Normalizzazione
  const float tScale = 30.0f, hScale = 4414.5f;
  auto tn = tempo / tScale;
  auto hn = altezza / hScale;
  torch::manual_seed(42);

  torch::nn::Sequential model_2(
    torch::nn::Linear(1, 32),
    torch::nn::Tanh(),            // più adatta di ReLU a una curva liscia
    torch::nn::Linear(32, 32),
    torch::nn::Tanh(),
    torch::nn::Linear(32, 1));

  torch::optim::Adam optimizer_2(model->parameters(), torch::optim::AdamOptions(learningRate));
  torch::Tensor tempo_pred_2;
  torch::Tensor loss_2;
  for (epoch = 0; epoch < epocs; ++epoch) {
    optimizer_2.zero_grad();
    tempo_pred_2 = model_2->forward(hn);
    //torch::Tensor loss = loss_func(tempo_pred, tempo);
    loss_2 = loss_func(tempo_pred_2, tn);
    loss_2.backward();
    optimizer_2.step();

    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << ", loss: " << loss_2.item<float>() << std::endl;
    }
  }
  std::cout << "Epoch " << epoch << ", loss: " << loss_2.item<float>() << std::endl;

  // Equivalente del plot: stampa dati reali e predizioni finali
  std::cout << "altezza | tempo | tempo_pred\n";
  auto pred_2 = tempo_pred_2.detach();
  for (int i = 0; i < hn.size(0); ++i) {
    std::cout << hn[i].item<float>() << " | "
              << tn[i].item<float>() << " | "
              << pred_2[i].item<float>() << "\n";
  }


  return 0;
}