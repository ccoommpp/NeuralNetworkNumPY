import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        self.layer_size = [784, 128, 64, 16, 10]
        self.memory = {}

    def init_layer(self,seed = 9):
        NN = []
        for i in range(1, len(self.layer_size)):
            input_dim = self.layer_size[i - 1]
            output_dim = self.layer_size[i]

            if i == len(self.layer_size) - 1:
                activation = "softmax"
            else:
                activation = "relu"

            layer ={
                "input_dim": input_dim,
                "output_dim": output_dim,
                "activation": activation
            }

            NN.append(layer)

        np.random.seed(seed)
        params_values = {}

        for idx,layer in enumerate(NN):
            layer_idx = idx+1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            params_values["W" + str(layer_idx)] = (np.random.randn(layer_output_size,layer_input_size) * np.sqrt(2 / layer_input_size))
            params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1

        return params_values,NN

    def softmax(self, Z):
        exp_values = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        exp_values_sum = np.sum(exp_values,axis=0,keepdims=True)
        return exp_values / exp_values_sum

    def relu(self,Z):
        return np.maximum(0,Z)

    def relu_backward(self,dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0;
        return dZ;

    def Foward_Propagation(self,X,param,NN):
        A = X
        self.memory["A0"] = X
        for i in range(1,len(NN)+1):
            W = param["W" + str(i)]
            B = param["b" + str(i)]
            activation = NN[i-1]["activation"]
            Z = np.dot(W , A) + B

            self.memory ["Z" + str(i)] = Z

            if activation == "relu":
                A = self.relu(Z)
            else:
                A = self.softmax(Z)
            self.memory ["A" + str(i)] = A

        return A

    def cross_entropy(self, Y_hat, Y):
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(Y_hat + 1e-8))/m
        return loss

    def one_hot(self,lables):
        return np.eye(10)[lables].T

    def back_propagation(self,Y_hat,Y,param,NN):
        m = Y.shape[1]
        dZ = Y_hat - Y
        grads={}

        for idx_prv in reversed(range(len(NN))):
            layer_idx_cur = idx_prv+1
            W = param["W" + str(layer_idx_cur)]

            A_prv = self.memory["A" +str(idx_prv)]
            W_cur = param["W" +str(layer_idx_cur)]

            dW_curr = np.dot(dZ, A_prv.T) / m
            db_curr = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(W_cur.T, dZ)
            if idx_prv > 0:
                Z_prev = self.memory["Z" + str(idx_prv)]
                dZ = self.relu_backward(dA_prev, Z_prev)
            grads["dW" + str(layer_idx_cur)] = dW_curr
            grads["db" + str(layer_idx_cur)] = db_curr

        for i in range(1, len(NN) + 1):
            param["W" + str(i)] -= 0.01 * grads["dW" + str(i)]
            param["b" + str(i)] -= 0.01 * grads["db" + str(i)]

        return param

    def train(self,X, Y, epochs):
        params_values, NN = self.init_layer()
        cost_history = []
        accuracy_history = []

        for i in range(epochs):
            print(i)
            Y_hat = self.Foward_Propagation(X, params_values, NN)
            cost = self.cross_entropy(Y_hat, Y)

            cost_history.append(cost)
            predictions = np.argmax(Y_hat, axis=0)
            actual = np.argmax(Y, axis=0)
            accuracy = np.mean(predictions == actual)
            accuracy_history.append(accuracy)

            params_values = self.back_propagation(Y_hat, Y , params_values, NN)

        return params_values, cost_history, accuracy_history


    def save_parameters(self, params, folder="model"):

        import os

        os.makedirs(folder, exist_ok=True)

        for name, value in params.items():

            np.savetxt(
                f"{folder}/{name}.csv",
                value,
                delimiter=","
            )

        print("Model saved.")

    def load_parameters(self, folder="model"):

        params = {}

        for i in range(1, 5):

            params["W" + str(i)] = np.loadtxt(
                f"{folder}/W{i}.csv",
                delimiter=","
            )

            params["b" + str(i)] = np.loadtxt(
                f"{folder}/b{i}.csv",
                delimiter=","
            )

            params["b" + str(i)] = params[
                "b" + str(i)
            ].reshape(-1, 1)

        return params

Network = NeuralNetwork()
df = pd.read_csv('mnist_test.csv')
df = np.array(df)
df = df.T

lable = df[0]
data = df[1:]/255.0

train_data = data[ : , :8000]
train_lable = lable[:8000]

test_data = data[ : , 8000:]
test_lable = lable[8000:]

Y_train = Network.one_hot(train_lable)
Y_test = Network.one_hot(test_lable)

trained_params, cost_history, accuracy_history = Network.train(train_data,Y_train,1000)
Network.save_parameters(trained_params,"model")

loaded_params = Network.load_parameters("model")

_,NN = Network.init_layer()
Y_test_hat = Network.Foward_Propagation(
    test_data,
    loaded_params,
    NN
)

predictions = np.argmax(
    Y_test_hat,
    axis=0
)

accuracy = np.mean(predictions == test_lable)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

plt.plot(cost_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

plt.plot(accuracy_history)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()
