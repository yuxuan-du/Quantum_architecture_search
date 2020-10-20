import pennylane as qml
from pennylane import numpy as np

import os
# load mnist data
data_all = np.load('test_binary_mnist_data.npy')
label_all = np.load('test_binary_mnist_label.npy')

print(data_all.shape)
print(label_all.shape)


data_train, label_train = data_all[:100], label_all[:100]
data_vali, label_vali = data_all[100:200], label_all[100:200]
data_test, label_test = data_all[200:300], label_all[200:300]
#np.random.seed(42)

# number of qubits in the circuit
nr_qubits = 3
# number of layers in the circuit
nr_layers = 3
encode_layers = 1
size = 100
# randomly initialize parameters from a normal distribution
params =  np.random.uniform(0, np.pi * 2, (nr_layers,    nr_qubits)) # I expand the trainable quantum gate in def layer
params = np.reshape(params, newshape = (nr_layers,    nr_qubits))
params_opt = np.zeros((nr_layers,     nr_qubits))
vali_acc_base = 0
#params = np.random.normal(0, np.pi, (nr_layers, nr_qubits))

def train_result_record():
    return {
            'loss': [],
            'train_acc': [],
            'valid_acc': [],
            'test_acc': []
            }

records = train_result_record()

# encoder
def encode_layer(feature, j):
    #qml.Hadamard(wires=0)
    #qml.Hadamard(wires=1)
    #qml.Hadamard(wires=2)

    for i in range(nr_qubits):
        qml.RY( feature[i], wires=i)
    
    phi = (np.pi - feature[0].val)*(np.pi - feature[1].val)*(np.pi - feature[2].val)
    # CRY 1
    #qml.CRY(phi, wires=[0, 1])
    # CRY 2
    #qml.CRY(phi, wires=[1, 2])

    

# a layer of the circuit ansatz
def layer(params, j):
    for i in range(nr_qubits):
        qml.RY(params[j,   i], wires=i)
        # qml.RZ(params[j, 2 * i], wires=i)
        # qml.RY(params[j, 2 * i + 1], wires=i)
        # qml.RZ(params[j, 2 * i + 2], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])


dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit( params, feature, A=None):
    # repeatedly apply each layer in the circuit
    #qml.templates.embeddings.AmplitudeEmbedding(features=feature, wires=range(2),   normalize=True )
    for j in range(encode_layers):
        encode_layer(feature, j)  # we use amplitude encoding to replace the kernel embedding
    for j in range(nr_layers):
        layer(params, j)
    return qml.expval(qml.Hermitian(A, wires=[0, 1]))


opt = qml.AdamOptimizer(0.05)
def cost_fn(params):
    global data_train, label_train
    loss = 0
    indices = np.arange(data_train.shape[0]) #np.random.permutation(data_train.shape[0])
    data_train = data_train[indices]
    label_train = label_train[indices]
    correct = 0
    for data, label in zip(data_train[:size], label_train[:size]):
        out = circuit( params, data, A=np.kron(np.eye(2), np.array([[1, 0], [0, 0]])))
        loss +=   (label - out)**2
        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    loss /= size
    print('loss: {} , acc: {} '.format(loss, correct / size))
    records['train_acc'].append(correct / size)
    records['loss'].append(loss._value)

    return loss

def test_fn(params):
    correct = 0
    for data, label in zip(data_test, label_test):
        out = circuit( params, data, A=np.kron(np.eye(2), np.array([[1, 0], [0, 0]])))

        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    print('Test acc: {}'.format(correct / label_test.shape[0]))
    records['test_acc'].append(correct / label_test.shape[0])

def valid_fn(params):
    correct = 0
    for data, label in zip(data_vali, label_vali):
        out = circuit( params, data, A=np.kron(np.eye(2), np.array([[1, 0], [0, 0]])))

        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    print('Valid acc: {}'.format(correct / label_vali.shape[0]))
    records['valid_acc'].append(correct / label_vali.shape[0])
    return correct / label_vali.shape[0]

for i in range(400):
    print('Epoch {}'.format(i))
    params = opt.step(cost_fn, params)
    # if (i+1) % 10 == 0:
    valid_acc = valid_fn(params)
    test_fn(params)
    if valid_acc > vali_acc_base:
        params_opt = params
    f = open('train_result' + '_noiselss'+ '.txt', 'w')
    f.write(str(records))
    f.close()


test_fn(params_opt)