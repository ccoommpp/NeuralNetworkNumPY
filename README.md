# NeuralNetworkNumPY

A fully connected neural network implemented from scratch using **NumPy**, trained on the **MNIST** dataset.

## Training Results

All experiments below were trained for **1000 epochs**.

---

## Training Accuracy

### Learning Rate: 0.01

<img src="stats/ACCURACY_01.png" width="650">

### Learning Rate: 0.001

<img src="stats/ACCURACY_001_1000(EPOCS).png" width="650">

### Learning Rate: 0.005

<img src="stats/ACCURACY_005.png" width="650">

---

## Training Loss

### Learning Rate: 0.01

<img src="stats/LOSS_01.png" width="650">

### Learning Rate: 0.001

<img src="stats/LOSS_001_1000(EPOCS).png" width="650">

### Learning Rate: 0.005

<img src="stats/LOSS_005.png" width="650">

---

## Implementation

The neural network is implemented entirely using NumPy.

- Forward propagation
- Backpropagation
- ReLU activation
- Softmax output
- Cross-entropy loss
- Gradient descent
- MNIST classification
- Training and loss visualization

## Project Structure

```text
NeuralNetworkNumPY/
│
├── NN.py
├── mnist_test.csv
├── README.md
│
└── stats/
    ├── ACCURACY_01(1).png
    ├── ACCURACY_001_1000(EPOCS).png
    ├── ACCURACY_005.png
    ├── ACCURACYSCORE_001.png
    ├── ACCURACYSCORE_005.png
    ├── LOSS_01.png
    ├── LOSS_001_1000(EPOCS).png
    └── LOSS_005.png
