# Comparative Analysis of Different CNN Architectures

## 📌 Problem Statement

The objective of this lab is to implement, train, and evaluate multiple landmark **Convolutional Neural Network (CNN)** architectures in order to study the impact of **network depth**, **architecture design**, and **dataset complexity** on classification performance and computational efficiency.

The experiments are conducted on standard benchmark datasets and include analysis of different **loss functions**, **optimizers**, and **feature-space visualizations**.

---

## 🔹 Part 1: Comparison of CNN Architectures

In this part, the following CNN architectures are implemented and evaluated:

- LeNet-5  
- AlexNet  
- VGGNet  
- ResNet-50  
- ResNet-100  
- EfficientNet  
- InceptionV3  
- MobileNet  

### 📂 Dataset Used
One of the following benchmark datasets is selected for experimentation:

- MNIST  
- Fashion-MNIST  
- CIFAR-10  

### 🎯 Objective
- Analyze how **network depth** and **architectural complexity** affect classification accuracy.
- Compare **training time**, **parameter count**, and **computational efficiency**.
- Study performance differences between **simple datasets (MNIST)** and **complex datasets (CIFAR-10)**.
- Experiment with different training configurations (epochs, optimizers, batch sizes).

---

## 🔹 Part 2: Impact of Loss Functions and Optimization Strategies

This part focuses on understanding how different **loss functions** and **optimizers** influence model convergence and final accuracy on datasets of varying complexity.

### 📌 Datasets
- **MNIST** – simple and balanced dataset  
- **CIFAR-10** – complex and noisy dataset  

### 📌 Loss Functions Compared
- Binary Cross-Entropy (BCE)
- Focal Loss
- ArcFace Loss

### 📌 Experimental Configuration

| Model     | Optimizer | Epochs | Loss Function | Training Accuracy | Testing Accuracy |
|-----------|-----------|--------|---------------|-------------------|------------------|
| VGGNet    | Adam      | 10     | BCE           | Recorded          | Recorded         |
| AlexNet  | SGD       | 20     | Focal Loss    | Recorded          | Recorded         |
| ResNet   | Adam      | 15     | ArcFace       | Recorded          | Recorded         |

### 🎯 Objective
- Study convergence behavior of different loss functions.
- Compare generalization ability on simple vs. complex datasets.
- Observe robustness of advanced loss functions in noisy classification tasks.

---

## 🔹 Part 3: Feature Space Visualization

To better understand how different loss functions influence feature learning, **feature-space visualizations** are performed.

### 📊 Visualization Techniques
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Decision boundary analysis (where applicable)

### 🎯 Objective
- Visualize how loss functions such as **BCE** and **ArcFace** cluster feature representations.
- Compare class separability on **CIFAR-10**.
- Analyze discriminative power of learned embeddings.

---

## 🧠 Key Learning Outcomes

- Understanding the trade-off between **model depth and performance**.
- Insights into **computational efficiency vs. accuracy**.
- Practical comparison of **advanced loss functions**.
- Visualization-based interpretation of CNN feature representations.

---

## 🛠 Tools & Frameworks

- Python
- TensorFlow / PyTorch
- NumPy, Matplotlib
- Google Colab / Jupyter Notebook

---

## 📎 Notes

All experiments were conducted as part of an academic lab assignment.  
Hyperparameters and configurations were selected to balance performance and training feasibility.

