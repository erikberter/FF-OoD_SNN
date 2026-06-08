# Forward-Forward Learning achieves Highly Selective Latent Representations for Out-of-Distribution Detection in Fully Spiking Neural Networks

- Erik B. Terres-Escudero - erik.terres@tecnalia.com
- Javier Del Ser 
- Aitor Martinez-Seras
- Pablo Garcia Bringas

## Abstract

In recent years, Artificial Intelligence (AI) models have achieved remarkable success across various domains, yet challenges persist in two critical areas: ensuring robustness against uncertain inputs and drastically increasing model efficiency during training and inference. Spiking Neural Networks (SNNs), inspired by biological systems, offer a promising avenue for overcoming these limitations. By operating in an event-driven manner, SNNs achieve low energy consumption and can naturally implement biological methods known for their high noise tolerance. In this work, we explore the potential of the spiking Forward-Forward Algorithm (FFA) to address these challenges, leveraging its representational properties for both Out-of-Distribution (OoD) detection and interpretability. To achieve this, we exploit the sparse and highly specialized neural latent space of FF networks to estimate the likelihood of a sample belonging to the training distribution. Additionally, we propose a novel, gradient-free attribution method to detect features that drive a sample away from class distributions, addressing the challenges posed by the lack of gradients in most visual interpretability methods for spiking models. We evaluate our OoD detection algorithm on well-known image datasets (e.g., Omniglot, Not-MNIST, CIFAR10), achieving an average AUROC of 92.30 on grayscale and 83.78 on colored datasets, and outperforming previous methods proposed in the literature for OoD detection in SNNs. Furthermore, our attribution method precisely identifies salient OoD features, such as artifacts or missing regions, hence providing a visual explanatory interface for the user to understand why unknown inputs are identified as such by the proposed method.


![image](img/pdf_Latents_F.pdf)

## Requirements

- Python 3.11
- PyTorch-compatible CUDA setup (for GPU runs)

## How to run

1. Create and activate a Python 3.11 environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Run training (example: MNIST, ANN):

```bash
python train.py --dataset "mnist" --neurons_per_layer "1400" --input_size "784" --batch_size "512" --threshold "6" --negative_threshold "2" --alpha "1" --beta "1" --epochs "10" --lr "0.001"
```

3. Run training (example: MNIST, SNN):

```bash
python train.py --dataset "mnist" --neurons_per_layer "1400" --input_size "784" --batch_size "512" --threshold "6" --negative_threshold "2" --alpha "1" --beta "1" --epochs "10" --lr "0.002" --use_snn
```

4. Run OOD evaluation:

```bash
python OOD_tests.py --dataset_group "small"
python OOD_tests.py --dataset_group "large" --feature_extractor "ResNet"
```

Outputs are written under experiments/.
