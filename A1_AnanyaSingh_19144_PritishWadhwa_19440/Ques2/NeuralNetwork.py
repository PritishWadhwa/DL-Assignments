# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from toolkit import MyNeuralNetwork, lossVsEpochPlot, accVsEpochPlot

# %%
trainData = pd.read_csv('./Data/fashion-mnist_train.csv')
testData = pd.read_csv('./Data/fashion-mnist_test.csv')

# %%
trainData.shape, testData.shape

# %%
trainData = trainData.values
testData = testData.values

# %%
trainX = trainData[:, 1:]
trainY = trainData[:, 0]
testX = testData[:, 1:]
testY = testData[:, 0]

# %%
trainX.shape, trainY.shape, testX.shape, testY.shape

# %%
# To visualize this, I have to convert it to a 2D array of size 28 x 28


def drawImg(X, Y, i):
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title("Label " + str(Y[i]))
    plt.show()


# %%
for i in range(10):
    drawImg(trainX, trainY, i)

# %%
normalize = StandardScaler()
trainX = normalize.fit_transform(trainX)
testX = normalize.transform(testX)

# %% [markdown]
# # Basic MLP

# %% [markdown]
# ### ReLU

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=2,
                        Layer_sizes=[200, 50], activation="relu",
                        learning_rate=0.1, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp2LayerLr0.1Relu",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=2,
                        Layer_sizes=[200, 50], activation="relu",
                        learning_rate=0.01, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp2LayerLr0.01Relu",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=2,
                        Layer_sizes=[200, 50], activation="relu",
                        learning_rate=0.001, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp2LayerLr0.001Relu",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[100], activation="relu",
                        learning_rate=0.1, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1LayerLr0.1Relu",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[100], activation="relu",
                        learning_rate=0.01, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1LayerLr0.01Relu",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[100], activation="relu",
                        learning_rate=0.001, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1LayerLr0.001Relu",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="relu",
                        learning_rate=0.1, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1Layer256Lr0.1relu",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="relu",
                        learning_rate=0.01, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1Layer256Lr0.01relu",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="relu",
                        learning_rate=0.001, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1Layer256Lr0.001relu",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %% [markdown]
# ### Sigmoid

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=2,
                        Layer_sizes=[200, 50], activation="sigmoid",
                        learning_rate=0.1, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp2LayerLr0.1sigmoid",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=2,
                        Layer_sizes=[200, 50], activation="sigmoid",
                        learning_rate=0.01, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp2LayerLr0.01sigmoid",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=2,
                        Layer_sizes=[200, 50], activation="sigmoid",
                        learning_rate=0.001, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp2LayerLr0.001sigmoid",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[100], activation="sigmoid",
                        learning_rate=0.1, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1LayerLr0.1sigmoid",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[100], activation="sigmoid",
                        learning_rate=0.01, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1LayerLr0.01sigmoid",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[100], activation="sigmoid",
                        learning_rate=0.001, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1LayerLr0.001sigmoid",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="sigmoid",
                        learning_rate=0.1, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1Layer256Lr0.1sigmoid",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="sigmoid",
                        learning_rate=0.01, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1Layer256Lr0.01sigmoid",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="sigmoid",
                        learning_rate=0.001, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1Layer256Lr0.001sigmoid",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %% [markdown]
# ### Tanh

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=2,
                        Layer_sizes=[200, 50], activation="tanh",
                        learning_rate=0.1, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp2LayerLr0.1tanh",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=2,
                        Layer_sizes=[200, 50], activation="tanh",
                        learning_rate=0.01, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp2LayerLr0.01tanh",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=2,
                        Layer_sizes=[200, 50], activation="tanh",
                        learning_rate=0.001, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp2LayerLr0.001tanh",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[100], activation="tanh",
                        learning_rate=0.1, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1LayerLr0.1tanh",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[100], activation="tanh",
                        learning_rate=0.01, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1LayerLr0.01tanh",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[100], activation="tanh",
                        learning_rate=0.001, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1LayerLr0.001tanh",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="tanh",
                        learning_rate=0.1, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1Layer256Lr0.1tanh",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="tanh",
                        learning_rate=0.01, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1Layer256Lr0.01tanh",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="tanh",
                        learning_rate=0.001, weight_init="random",
                        num_epochs=150, batch_size=len(trainX),
                        backpropogation='gd', modelName="./Models/mlp1Layer256Lr0.001tanh",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %% [markdown]
# ## MLP with Different Optimizers

# %% [markdown]
# ### Gradient Descent with Momentum

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="relu",
                        learning_rate=0.0001, weight_init="random",
                        num_epochs=50, batch_size=64,
                        backpropogation='momentum', modelName="./Models/mlp1Layer256Lr0.1reluMomentum",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %% [markdown]
# ### Nesterov's Accelerated Gradient
#

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="relu",
                        learning_rate=0.0001, weight_init="random",
                        num_epochs=50, batch_size=64,
                        backpropogation='nag', modelName="./Models/mlp1Layer256Lr0.1reluNag",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %% [markdown]
# ### Adagrad

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="relu",
                        learning_rate=0.0001, weight_init="random",
                        num_epochs=50, batch_size=64,
                        backpropogation='adagrad', modelName="./Models/mlp1Layer256Lr0.1reluAdagrad",
                        loadModel=False, saveModel=False)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %% [markdown]
# ### RMSProp

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="relu",
                        learning_rate=0.0001, weight_init="random",
                        num_epochs=50, batch_size=64,
                        backpropogation='rmsprop', modelName="./Models/mlp1Layer256Lr0.1reluRMSProp",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))

# %% [markdown]
# ### Adam

# %%
model = MyNeuralNetwork(N_inputs=784, N_outputs=10, N_layers=1,
                        Layer_sizes=[256], activation="relu",
                        learning_rate=0.0001, weight_init="random",
                        num_epochs=50, batch_size=64,
                        backpropogation='adam', modelName="./Models/mlp1Layer256Lr0.1reluAdam",
                        loadModel=False, saveModel=True)
trainLoss, testLoss, trainAccs, testAccs = model.fit(
    trainX, trainY, validX=testX, validY=testY, logs=True)
lossVsEpochPlot(trainLoss, testLoss)
accVsEpochPlot(trainAccs, testAccs)
print("Test Accuracy: " + str(model.score(testX, testY)))
