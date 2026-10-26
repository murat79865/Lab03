# Lab 3 — Architectures and Their Costs

## 1. Introduction

The goal of this laboratory assignment is to compare three neural network architectures: Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Attention-based model.

The models were evaluated using the Fashion-MNIST dataset.

The comparison focuses on:
- Number of model parameters
- Test accuracy
- Inference latency
- Computational efficiency
- Differences in data access patterns

## 2. Dataset

The Fashion-MNIST dataset contains grayscale images of clothing items.

Each image has a resolution of 28 × 28 pixels.

The dataset was divided into:

| Dataset | Number of images |
|---|---:|
| Training | 55,000 |
| Validation | 5,000 |
| Testing | 10,000 |

The batch size was set to 128.

The images were converted to tensors and normalized using:

- Mean: 0.2860
- Standard deviation: 0.3530

## 3. Experimental Environment

The models were implemented using Python and PyTorch.

The experiments were performed on a CPU.

The models were trained using:

- Optimizer: Adam
- Learning rate: 0.001
- Number of epochs: 5
- Loss function: CrossEntropyLoss

The inference latency was measured for batch sizes 1 and 128.

The median latency was calculated using repeated measurements.

Model parameters were inspected using torchinfo.

Computational cost was estimated using THOP.

## 4. Experimental Results

### 4.1. Model Parameters

The number of trainable parameters was obtained using PyTorch model inspection.

| Model | Parameters |
|---|---:|
| MLP | 235,146 |
| CNN | 20,490 |
| Attention | 70,794 |

The MLP has the largest number of parameters among the three models.

The CNN has the smallest number of parameters.

### 4.2. Test Accuracy

The models were trained for five epochs and evaluated on the Fashion-MNIST test dataset.

| Model | Test Accuracy |
|---|---:|
| MLP | 87.52% |
| CNN | 89.65% |
| Attention | 83.83% |

The CNN achieved the highest test accuracy of 89.65%.

The Attention model achieved 83.83%, while the MLP achieved 87.52%.

### 4.3. Inference Latency

Inference latency was measured on the CPU.

| Model | Batch Size 1 (ms) | Batch Size 128 (ms) |
|---|---:|---:|
| MLP | 0.0713 | 0.2511 |
| CNN | 0.1676 | 7.2368 |
| Attention | 0.6220 | 2.8753 |

For batch size 1, the MLP achieved the lowest latency.

For batch size 128, the MLP also achieved the lowest latency.

The Attention model had the highest latency for batch size 1.

The CNN had the highest latency for batch size 128.

### 4.4. Computational Cost (MACs and FLOPs)

The computational cost of the models was estimated using THOP.

One MAC represents one multiplication and accumulation operation.

For this comparison, one MAC is counted as approximately two FLOPs.

| Model | MACs (THOP) | Estimated FLOPs |
|---|---:|---:|
| MLP | 234,752 | 469,504 |
| CNN | 1,031,744 | 2,063,488 |
| Attention | 591,488 | 1,182,976 |

#### Manual FLOPs Calculation

For the MLP, the number of MACs is calculated from the fully connected layers:

784 × 256 + 256 × 128 + 128 × 10 = 234,752 MACs

Estimated FLOPs:

234,752 × 2 = 469,504 FLOPs

For the CNN, the convolutional and fully connected layers are considered.

First convolution:

28 × 28 × 16 × 1 × 3 × 3 = 112,896 MACs

Second convolution:

14 × 14 × 32 × 16 × 3 × 3 = 903,168 MACs

Final fully connected layer:

1,568 × 10 = 15,680 MACs

Total:

112,896 + 903,168 + 15,680 = 1,031,744 MACs

Estimated FLOPs:

1,031,744 × 2 = 2,063,488 FLOPs

The manual calculations for the MLP and CNN match the THOP results.

The FLOPs values are estimates based on multiplication and accumulation operations. Bias additions, activation functions, pooling operations, and some Transformer operations are not included in these manual calculations.

THOP may not count all Transformer operations and parameters. Therefore, the Attention parameter count reported by torchinfo is used in this report.

## 5. Graphical Results

### 5.1. Test Accuracy vs Model Parameters

![Test Accuracy vs Model Parameters](accuracy_vs_parameters.png)

This graph compares the test accuracy of the models with their number of parameters.

The CNN achieved the highest accuracy with the lowest number of parameters.

### 5.2. Test Accuracy Comparison

![Fashion-MNIST Test Accuracy](model_accuracy.png)

The CNN achieved the highest test accuracy of 89.65%.

The MLP achieved 87.52%, while the Attention model achieved 83.83%.

### 5.3. Inference Latency — Batch Size 1

![Inference Latency Batch 1](latency_batch1.png)

The MLP achieved the lowest inference latency for batch size 1.

The Attention model had the highest latency.

### 5.4. Inference Latency — Batch Size 128

![Inference Latency Batch 128](latency_batch128.png)

The MLP achieved the lowest inference latency for batch size 128.

The CNN had the highest latency in this experiment.

## 6. Architecture and Access Patterns

### 6.1. MLP

The MLP processes the input image as a flattened vector of 784 values.

It uses fully connected layers to learn relationships between pixels.

The MLP has a global access pattern because each neuron in a fully connected layer is connected to all neurons in the previous layer.

This architecture is simple and achieved low inference latency in our experiment.

### 6.2. CNN

The CNN uses convolutional layers to extract local features from images.

Convolutional filters process small neighboring regions of the image.

The CNN uses weight sharing, meaning the same filter is applied across different image locations.

This reduces the number of parameters compared with a fully connected architecture.

The CNN achieved the highest test accuracy in our experiment.

### 6.3. Attention

The Attention model divides the image into patches and processes their relationships using Transformer encoder layers.

Self-attention allows the model to compare different patches.

This provides a global access pattern because each patch can interact with other patches.

The Attention model had higher inference latency for batch size 1 in our experiment.

## 7. Conclusion

This laboratory assignment compared MLP, CNN, and Attention-based architectures using the Fashion-MNIST dataset.

The CNN achieved the highest test accuracy of 89.65%.

The MLP achieved the lowest inference latency for both batch sizes.

The Attention model demonstrated the ability to process relationships between image patches, but its inference latency was higher than the MLP.

The experiment shows that model performance depends not only on the number of parameters but also on the architecture and its computational operations.

Overall, the CNN provided the best test accuracy, while the MLP provided the fastest inference in our measurements.