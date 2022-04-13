import numpy as np
import matplotlib.pyplot as plt
import pickle

class MyNeuralNetwork:
    # Custom implementation of Neural Network Class
    def __init__(self, N_inputs, N_outputs, N_layers=2, Layer_sizes=[10, 5], activation="sigmoid", learning_rate=0.1, weight_init="random", batch_size=1, num_epochs=200, backpropogation='gd', beta=0.9, gamma=0.999, modelName=None, loadModel=False, saveModel=True):
        """
        N_inputs: input size
        N_outputs: outputs size
        N_layers: number of hidden layers
        Layer_sizes: list of hidden layer sizes
        activation: activation function to be used (ReLu, Leaky ReLu, sigmoid, linear, tanh, softmax)
        learning_rate: learning rate
        weight_init: weight initialization (zero, random, normal)
        batch_size: batch size
        num_epochs: number of epochs
        """
        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        self.N_layers = N_layers
        self.Layer_sizes = Layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.backpropogation = backpropogation
        self.beta = beta
        self.gamma = gamma
        self.modelSave = saveModel
        self.modelName = modelName
        self.iterations = 1
        np.random.seed(0)

        model = {}
        if weight_init == "zero":
            model['W1'] = np.zeros((N_inputs, Layer_sizes[0]))
            model['b1'] = np.zeros((1, Layer_sizes[0]))
            for i in range(1, N_layers):
                model['W' +
                      str(i+1)] = np.zeros((Layer_sizes[i-1], Layer_sizes[i]))
                model['b' + str(i+1)] = np.zeros((1, Layer_sizes[i]))
            model['W' + str(N_layers+1)
                  ] = np.zeros((Layer_sizes[-1], N_outputs))
            model['b' + str(N_layers+1)] = np.zeros((1, N_outputs))
        elif weight_init == "random":
            model['W1'] = np.random.randn(N_inputs, Layer_sizes[0])*0.01
            model['b1'] = np.zeros((1, Layer_sizes[0]))
            for i in range(1, N_layers):
                model['W' + str(i+1)] = np.random.randn(Layer_sizes[i-1],
                                                        Layer_sizes[i])*0.01
                model['b' + str(i+1)] = np.zeros((1, Layer_sizes[i]))
            model['W' + str(N_layers+1)
                  ] = np.random.randn(Layer_sizes[-1], N_outputs)*0.01
            model['b' + str(N_layers+1)] = np.zeros((1, N_outputs))
        elif weight_init == "normal":
            model['W1'] = np.random.normal(
                0, 1, (N_inputs, Layer_sizes[0]))*0.01
            model['b1'] = np.zeros((1, Layer_sizes[0]))
            for i in range(1, N_layers):
                model['W' + str(i+1)] = np.random.normal(0, 1,
                                                         (Layer_sizes[i-1], Layer_sizes[i]))*0.01
                model['b' + str(i+1)] = np.zeros((1, Layer_sizes[i]))
            model['W' + str(N_layers+1)] = np.random.normal(0,
                                                            1, (Layer_sizes[-1], N_outputs))*0.01
            model['b' + str(N_layers+1)] = np.zeros((1, N_outputs))
        else:
            print("Invalid weight initialization")
            return
        
        if loadModel:
            model = self.loadModel()

        self.model = model
        self.activationOutputs = None

    def relu_forward(self, X):
        """
        ReLu activation function for forward propagation
        X: input
        return: output after applying the relu function
        """
        return np.maximum(X, 0)

    def relu_backward(self, X):
        """
        ReLu activation function for backpropagation
        X: input
        return: output after applying the gradient of relu function
        """
        return np.where(X > 0, 1, 0)

    def sigmoid_forward(self, X):
        """
        Sigmoid activation function
        X: input
        return: output after applying the sigmoid function
        """
        return 1/(1+np.exp(-X))

    def sigmoid_backward(self, X):
        """
        Sigmoid activation function
        X: input
        return: output after applying the gradient of sigmoid function
        """
        return self.sigmoid_forward(X)*(1-self.sigmoid_forward(X))
        # return X*(1-X)

    def tanh_forward(self, X):
        """
        Tanh activation function
        X: input
        return: output after applying the tanh function
        """
        return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
        # return np.tanh(X)

    def tanh_backward(self, X):
        """
        Tanh activation function
        X: input
        return: output after applying the gradient of tanh function
        """
        return 1-(self.tanh_forward(X)**2)
        # return 1-X**2

    def softmax_forward(self, X):
        """
        Softmax activation function
        X: input
        return: output after applying the softmax function
        """
        exp = np.exp(X - np.max(X))
        return exp/np.sum(exp, axis=1, keepdims=True)

    def softmax_backward_actual(self, X):
        """
        Softmax activation function
        X: input
        return: output after applying the gradient of softmax function
        """
        s = self.softmax_forward(X).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def softmax_backward(self, X):
        """
        Softmax activation function
        X: input
        return: output after applying the gradient of softmax function
        """
        return self.softmax_forward(X)*(1-self.softmax_forward(X))

    def forward(self, X):
        """
        Forward propagation
        X: input
        return: output after applying the activation function
        """
        if self.activation == "relu":
            currentActivationFuntion = self.relu_forward
        elif self.activation == "sigmoid":
            currentActivationFuntion = self.sigmoid_forward
        elif self.activation == "tanh":
            currentActivationFuntion = self.tanh_forward
        elif self.activation == "softmax":
            currentActivationFuntion = self.softmax_forward
        else:
            raise ValueError("Invalid activation function")

        self.activationOutputs = {}

        self.activationOutputs['Z1'] = np.dot(
            X, self.model['W1']) + self.model['b1']
        self.activationOutputs['A1'] = currentActivationFuntion(
            self.activationOutputs['Z1'])
        # self.activationOutputs['A1'] = np.tanh(self.activationOutputs['Z1'])

        for i in range(2, self.N_layers+1):
            self.activationOutputs['Z' + str(i)] = np.dot(self.activationOutputs['A' + str(
                i-1)], self.model['W' + str(i)]) + self.model['b' + str(i)]
            self.activationOutputs['A' + str(i)] = currentActivationFuntion(
                self.activationOutputs['Z' + str(i)])

        self.activationOutputs['Z' + str(self.N_layers+1)] = np.dot(self.activationOutputs['A' + str(
            self.N_layers)], self.model['W' + str(self.N_layers+1)]) + self.model['b' + str(self.N_layers+1)]
        self.activationOutputs['A' + str(self.N_layers+1)] = self.softmax_forward(
            self.activationOutputs['Z' + str(self.N_layers+1)])

        return self.activationOutputs['A' + str(self.N_layers+1)]

    def backward(self, X, Y):
        """
        Backward propagation
        X: input
        Y: output
        """
        if self.activation == "relu":
            currentActivationFuntion = self.relu_backward
        elif self.activation == "sigmoid":
            currentActivationFuntion = self.sigmoid_backward
        elif self.activation == "tanh":
            currentActivationFuntion = self.tanh_backward
        elif self.activation == "softmax":
            currentActivationFuntion = self.softmax_backward
        else:
            raise ValueError("Invalid activation function")

        # computing the gradients
        self.gradients = {}
        self.gradients['delta' + str(self.N_layers+1)] = (
            self.activationOutputs['A' + str(self.N_layers+1)] - Y)
        self.gradients['dW' + str(self.N_layers+1)] = (1/len(X)) * np.dot(self.activationOutputs['A' + str(
            self.N_layers)].T, self.gradients['delta' + str(self.N_layers+1)])
        self.gradients['db' + str(self.N_layers+1)] = (1/len(X)) * np.sum(
            self.gradients['delta' + str(self.N_layers+1)], axis=0, keepdims=True)

        for i in range(self.N_layers, 1, -1):
            self.gradients['delta' + str(i)] = np.dot(self.gradients['delta' + str(i+1)], self.model['W' + str(
                i+1)].T) * currentActivationFuntion(self.activationOutputs['Z' + str(i)])
            self.gradients['dW' + str(i)] = (1/len(X)) * np.dot(
                self.activationOutputs['A' + str(i-1)].T, self.gradients['delta' + str(i)])
            self.gradients['db' + str(i)] = (1/len(X)) * np.sum(
                self.gradients['delta' + str(i)], axis=0, keepdims=True)

        self.gradients['delta1'] = np.dot(
            self.gradients['delta2'], self.model['W2'].T) * currentActivationFuntion(self.activationOutputs['Z1'])
        self.gradients['dW1'] = (1/len(X)) * \
            np.dot(X.T, self.gradients['delta1'])
        self.gradients['db1'] = (
            1/len(X)) * np.sum(self.gradients['delta1'], axis=0, keepdims=True)

        # updating the model parameters
        if self.backpropogation == 'gd':
            for i in range(1, self.N_layers+2):
                self.model['W' + str(i)] -= self.learning_rate * \
                    self.gradients['dW' + str(i)]
                self.model['b' + str(i)] -= self.learning_rate * \
                    self.gradients['db' + str(i)]

        elif self.backpropogation == 'momentum':
            for i in range(1, self.N_layers+2):
                if ('v_W' + str(i)) not in self.model:
                    self.model['v_W' + str(i)] = 0
                if ('v_b' + str(i)) not in self.model:
                    self.model['v_b' + str(i)] = 0
                self.model['v_W' + str(i)] = (self.beta*self.model['v_W' + str(i)]) - (self.learning_rate * self.gradients['dW' + str(i)])
                self.model['v_b' + str(i)] = (self.beta*self.model['v_b' + str(i)]) - (self.learning_rate * self.gradients['db' + str(i)])
                self.model['W' + str(i)] = self.model['W' + str(i)] + self.model['v_W' + str(i)]
                self.model['b' + str(i)] = self.model['b' + str(i)] + self.model['v_b' + str(i)]

        elif self.backpropogation == 'nag':
            for i in range(1, self.N_layers+2):
                if ('v_W' + str(i)) not in self.model:
                    self.model['v_W' + str(i)] = 0
                if ('v_b' + str(i)) not in self.model:
                    self.model['v_b' + str(i)] = 0
                prevV_W = self.model['v_W' + str(i)]
                prevV_b = self.model['v_b' + str(i)]
                self.model['v_W' + str(i)] = (self.beta*self.model['v_W' + str(i)]) - (self.learning_rate * self.gradients['dW' + str(i)])
                self.model['v_b' + str(i)] = (self.beta*self.model['v_b' + str(i)]) - (self.learning_rate * self.gradients['db' + str(i)])
                self.model['W' + str(i)] = self.model['W' + str(i)] + ((1 + self.beta)*self.model['v_W' + str(i)] - self.beta*prevV_W)
                self.model['b' + str(i)] = self.model['b' + str(i)] + ((1 + self.beta)*self.model['v_b' + str(i)] - self.beta*prevV_b)
        
        elif self.backpropogation == 'adagrad':
            epsilon = 0.0000001
            for i in range(1, self.N_layers+2):
                if ('W_squaredSum' + str(i)) not in self.model:
                    self.model['W_squaredSum' + str(i)] = [0]
                if ('b_squaredSum' + str(i)) not in self.model:
                    self.model['b_squaredSum' + str(i)] = [0]
                self.model['W_squaredSum' + str(i)].append(self.model['W_squaredSum' + str(i)][-1] + np.square(self.gradients['dW' + str(i)]))
                self.model['b_squaredSum' + str(i)].append(self.model['b_squaredSum' + str(i)][-1] + np.square(self.gradients['db' + str(i)]))
                self.model['W' + str(i)] = self.model['W' + str(i)] - (self.learning_rate*self.gradients['dW' + str(i)])/np.sqrt(self.model['W_squaredSum' + str(i)][-1] + epsilon)
                self.model['b' + str(i)] = self.model['b' + str(i)] - (self.learning_rate*self.gradients['db' + str(i)])/np.sqrt(self.model['b_squaredSum' + str(i)][-1] + epsilon)

        elif self.backpropogation == 'rmsprop':
            epsilon = 0.0000001
            for i in range(1, self.N_layers+2):
                if ('v_W' + str(i)) not in self.model:
                    self.model['v_W' + str(i)] = 0
                if ('v_b' + str(i)) not in self.model:
                    self.model['v_b' + str(i)] = 0
                self.model['v_W' + str(i)] = self.beta*self.model['v_W' + str(i)] + (1-self.beta)*np.square(self.gradients['dW' + str(i)])
                self.model['v_b' + str(i)] = self.beta*self.model['v_b' + str(i)] + (1-self.beta)*np.square(self.gradients['db' + str(i)])
                self.model['W' + str(i)] = self.model['W' + str(i)] - (self.learning_rate*self.gradients['dW' + str(i)])/np.sqrt(self.model['v_W' + str(i)] + epsilon)
                self.model['b' + str(i)] = self.model['b' + str(i)] - (self.learning_rate*self.gradients['db' + str(i)])/np.sqrt(self.model['v_b' + str(i)] + epsilon)

        elif self.backpropogation == 'adam':
            epsilon = 0.0000001
            for i in range(1, self.N_layers+2):
                if ('v_W' + str(i)) not in self.model:
                    self.model['v_W' + str(i)] = 0
                if ('v_b' + str(i)) not in self.model:
                    self.model['v_b' + str(i)] = 0
                if ('s_W' + str(i)) not in self.model:
                    self.model['s_W' + str(i)] = 0
                if ('s_b' + str(i)) not in self.model:
                    self.model['s_b' + str(i)] = 0
                self.model['v_W' + str(i)] = self.beta*self.model['v_W' + str(i)] - (1-self.beta)*self.gradients['dW' + str(i)]
                self.model['v_b' + str(i)] = self.beta*self.model['v_b' + str(i)] - (1-self.beta)*self.gradients['db' + str(i)]
                self.model['s_W' + str(i)] = self.gamma*self.model['s_W' + str(i)] + (1 - self.gamma)*np.square(self.gradients['dW' + str(i)])
                self.model['s_b' + str(i)] = self.gamma*self.model['s_b' + str(i)] + (1 - self.gamma)*np.square(self.gradients['db' + str(i)])
                v_W = self.model['v_W' + str(i)]/(1-np.power(self.beta, self.iterations))
                v_b = self.model['v_b' + str(i)]/(1-np.power(self.beta, self.iterations))
                s_W = self.model['s_W' + str(i)]/(1-np.power(self.gamma, self.iterations)) 
                s_b = self.model['s_b' + str(i)]/(1-np.power(self.gamma, self.iterations))
                # v_W = self.model['v_W' + str(i)]
                # v_b = self.model['v_b' + str(i)]
                # s_W = self.model['s_W' + str(i)]
                # s_b = self.model['s_b' + str(i)]
                # print(np.sqrt(s_W), np.sqrt(s_b))
                self.model['W' + str(i)] = self.model['W' + str(i)] + (self.learning_rate*v_W)/(np.sqrt(s_W + epsilon))
                self.model['b' + str(i)] = self.model['b' + str(i)] + (self.learning_rate*v_b)/(np.sqrt(s_b + epsilon))

        else:
            raise ValueError("Invalid backpropogation algorithm")

    def oneHotEncoder(self, y, n_classes):
        """
        One hot encoder
        y: input
        return: encoded output
        """
        m = y.shape[0]
        y_oht = np.zeros((m, n_classes))
        y_oht[np.arange(m), y] = 1
        return y_oht

    def crossEntropyLoss(self, y_oht, y_prob):
        """
        Cross entropy loss
        y_oht: one hot encoded output
        y_prob: probabilities for classes
        return: cross entropy loss
        """
        return -np.mean(y_oht * np.log(y_prob + 1e-8))

    def fit(self, X, y, validX=None, validY=None, logs=True):
        """
        Fit the model to the data
        X: input
        Y: output
        epochs: number of epochs
        """
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        classes = self.N_outputs
        batchSize = self.batch_size
        y_oht = self.oneHotEncoder(y, classes)
        if validX is not None and validY is not None:
            y_oht_valid = self.oneHotEncoder(validY, classes)
        for i in range(self.num_epochs):
            for j in range(0, X.shape[0], batchSize):
                X_batch = X[j:j+batchSize]
                y_batch = y_oht[j:j+batchSize]
                y_ = self.forward(X_batch)
                self.backward(X_batch, y_batch)
            y_ = self.forward(X)
            train_loss = self.crossEntropyLoss(y_oht, y_)
            train_losses.append(train_loss)
            if validX is not None and validY is not None:
                y_valid = self.forward(validX)
                valid_loss = self.crossEntropyLoss(y_oht_valid, y_valid)
                valid_losses.append(valid_loss)
                validAcc = self.score(validX, validY)
                valid_accs.append(validAcc)
            trainAcc = self.score(X, y)
            train_accs.append(trainAcc)
            if logs:
                print("Epoch: {}, Loss: {}, Score: {}".format(
                    i, train_loss, trainAcc))
            self.iterations += 1
        if self.modelSave:
            self.saveModel()
        if validX is not None and validY is not None:
            return train_losses, valid_losses, train_accs, valid_accs
        return train_losses, train_accs

    def predict_proba(self, X):
        """
        Predict probabilities
        X: input
        return: probabilities
        """
        return self.forward(X)

    def predict(self, X):
        """
        Predict classes
        X: input
        return: classes
        """
        return np.argmax(self.forward(X), axis=1)

    def score(self, X, y):
        """
        Score the model
        X: input
        Y: output
        return: accuracy
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)*100

    def saveModel(self):
        """
        Save the weights
        filename: name of the file
        """
        model = {
            "name": self.modelName, 
            "model": self.model
        }
        with open(f'{self.modelName}.pickle', 'wb') as f:
            pickle.dump(model, f)

    def loadModel(self):
        """
        Load the weights
        filename: name of the file
        """
        with open(f'{self.modelName}.pickle', 'rb') as f:
            model = pickle.load(f)
            return model['model']
        #         self.model = np.load(filename, allow_pickle=True).item()
        

def lossVsEpochPlot(trainLoss, testLoss):
    """
    Plot the loss vs epoch plot
    trainLoss: train loss
    validLoss: test loss
    """
    plt.plot(trainLoss, label="Train Loss", color="blue")
    plt.plot(testLoss, label="Test Loss", color="red")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs Plot")
#     plt.savefig("Plots/Ques2/part1/{}_lossVsEpochs.png".format(funtion))
    plt.show()


def accVsEpochPlot(trainAcc, testAcc):
    """
    Plot the accuracy vs epoch plot
    trainAcc: train accuracy
    validAcc: test accuracy
    """
    plt.plot(trainAcc, label="Train Accuracy", color="blue")
    plt.plot(testAcc, label="Test Accuracy", color="red")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs for ")
#     plt.savefig("Plots/Ques2/part1/{}_accVsEpochs.png".format(funtion))
    plt.show()