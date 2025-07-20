import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("mnist_train.csv")


#print(data.head())


data = np.array(data)

a, b = data.shape
np.random.shuffle(data)

data_initial = data[0:1000].T
Y_dev = data_initial[0]
X_dev = data_initial[1:b]

X_dev = X_dev / 255  

print(X_dev)

data_fin = data[1000:a].T
Y_train = data_fin[0]
X_train = data_fin[1:b]
X_train = X_train / 255 
_,a_train =  X_train.shape


'''
print(m_shape)
print(X_train)
'''

#print(X_train[:, 0].shape)


def initial_parameters():

	Weight_1 = np.random.rand(10, 784) - 0.5
	bias_1 = np.random.rand(10, 1) - 0.5 
	Weight_2 = np.random.rand(10, 10) - 0.5 
	bias_2 = np.random.rand(10, 1) - 0.5 

	return Weight_1, bias_1, Weight_2, bias_2

def ReLU(Z):
	return np.maximum(Z, 0)

def derivative_of_Relu(Z):
	return Z > 0

def softmax(Z):
	A = np.exp(Z) / sum(np.exp(Z))
	return A

def for_prop(Weight_1, bias_1, Weight_2, bias_2, X):

	Z1 = Weight_1.dot(X) + bias_1
	A1 = ReLU(Z1)
	Z2 = Weight_2.dot(A1) + bias_2
	A2 = softmax(Z2)
	return Z1, A1, Z2, A2

def one_hot(Y):
	one_hot_Y = np.zeros((Y.size, Y.max() + 1))
	one_hot_Y[np.arange(Y.size), Y] = 1
	one_hot_Y = one_hot_Y.T
	return one_hot_Y

def back_prop(Z1, A1, Z2, A2, Weight_1, Weight_2, X, Y):
	#m = Y.size
	one_hot_Y = one_hot(Y)

	dZ2 = A2 - one_hot_Y
	dW2 = (1/a) * (dZ2.dot(A1.T))
	db2 = (1/a) * np.sum(dZ2)
	dZ1 = Weight_2.T.dot(dZ2) * derivative_of_Relu(Z1)
	dW1 = (1/a) * (dZ1.dot(X.T))
	db1 = (1/a) * np.sum(dZ1)

	return dW1, db1, dW2, db2

def update_parameters(Weight_1, bias_1, Weight_2, bias_2, dW1, db1, dW2, db2, P):

	Weight_1 = Weight_1 - P * dW1 
	bias_1 = bias_1 - P * db1 
	Weight_2 = Weight_2 - P * dW2 
	bias_2 = bias_2 - P * db2 

	return Weight_1, bias_1, Weight_2, bias_2

def get_preds(A2):
	return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
	print(predictions, Y)
	return np.sum(predictions == Y) / Y.size


def g_d(X, Y, P, iterations):

	Weight_1, bias_1, Weight_2, bias_2 = initial_parameters()
	for i in range(iterations):
		Z1, A1, Z2, A2 = for_prop(Weight_1, bias_1, Weight_2, bias_2, X)
		dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, Weight_1, Weight_2, X, Y)
		Weight_1, bias_1, Weight_2, bias_2 = update_parameters(Weight_1, bias_1, Weight_2, bias_2, dW1, db1, dW2, db2, P)
		if i % 10 == 0:
			print("Iterations : ", i)
			predictions = get_preds(A2)
			print("Accuracy : ", get_accuracy(predictions, Y))

	return Weight_1, bias_1, Weight_2, bias_2 


Weight_1, bias_1, Weight_2, bias_2 = g_d(X_train, Y_train, 0.1, 500)


def make_preds(X, Weight_1, bias_1, Weight_2, bias_2):
    _, _, _, A2 = for_prop(Weight_1, bias_1, Weight_2, bias_2, X)
    predictions = get_preds(A2)
    return predictions

def test_pred(index, Weight_1, bias_1, Weight_2, bias_2):
    current_image = X_train[:, index, None]
    prediction = make_preds(X_train[:, index, None], Weight_1, bias_1, Weight_2, bias_2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


'''
random_numbers = []

for _ in range(10):
    random_number = np.random.randint(0, 1001)
    random_numbers.append(random_number)

print(random_numbers)

'''
'''
for p in range(10):
	h = np.random.randint(0, 1001)
	test_pred(h, Weight_1, bias_1, Weight_2, bias_2)
'''	
'''
print(a)

print("/n")

print(b)

print("/n")

print(data_initial.T)

print("/n")

print(data_initial.T.shape)

'''