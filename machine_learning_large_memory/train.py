import argparse
import pennylane as qml
from pennylane import numpy as np
import time
import os
import json
from model import CircuitModel

parser = argparse.ArgumentParser("QAS")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=400, help='num of training epochs')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='', help='which architecture to use')
parser.add_argument('--noise', action='store_true', default=False, help='use noise')
# circuit
parser.add_argument('--n_qubits', type=int, default=3, help='number of qubits')
parser.add_argument('--n_encode_layers', type=int, default=1, help='number of encoder layers')
parser.add_argument('--n_layers', type=int, default=3, help='number of layers')

args = parser.parse_args()
args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(args.save)
with open(os.path.join(args.save, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.noise:
    import qiskit
    import qiskit.providers.aer.noise as noise


def cost_fn(params, model, inputs, labels, records):
    loss = 0
    correct = 0
    size = inputs.shape[0]
    model.params = params
    for data, label in zip(inputs, labels):
        out = model(data)
        loss += (label - out)**2
        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    loss /= size
    print('loss: {} , acc: {} '.format(loss._value, correct / size))
    records['train_acc'].append(correct / size)
    records['loss'].append(loss._value)
    return loss


def test_fn(model, inputs, labels):
    correct = 0
    for data, label in zip(inputs, labels):
        out = model(data)

        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    acc = correct / inputs.shape[0]
    return acc


def main():

    np.random.seed(args.seed)
    records = {
        'loss': [],
        'train_acc': [],
        'valid_acc': [],
        'test_acc': 0
    }
    '''init device'''
    if args.noise:
        # Error probabilities
        prob_1 = 0.05  # 1-qubit gate
        prob_2 = 0.2   # 2-qubit gate
        # Depolarizing quantum errors
        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2)
        # Add errors to noise model
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        print(noise_model)
        dev = qml.device('qiskit.aer', wires=args.n_qubits, noise_model=noise_model)
    else:
        dev = qml.device("default.qubit", wires=args.n_qubits)

    '''init model'''
    model = CircuitModel(dev, args.n_qubits, args.n_encode_layers, args.n_layers, args.arch)
    opt = qml.AdamOptimizer(0.05)

    '''load mnist data'''
    data_all = np.load(os.path.join(args.data, 'binary_mnist_data.npy'))
    label_all = np.load(os.path.join(args.data, 'binary_mnist_label.npy'))
    print('data size: {}'.format(data_all.shape))
    # split data
    data_train, label_train = data_all[:100], label_all[:100]
    data_valid, label_valid = data_all[100:200], label_all[100:200]
    data_test, label_test = data_all[200:300], label_all[200:300]

    '''train'''
    best_acc = 0
    best_params = None
    for epoch in range(args.epochs):
        model.params = opt.step(lambda params: cost_fn(params, model, data_train, label_train, records), model.params) # train
        acc = test_fn(model, data_valid, label_valid)
        records['valid_acc'].append(acc)
        print('Epoch: {}, valid acc: {}'.format(epoch, acc))
        if acc > best_acc:
            best_acc = acc
            best_params = model.params
            np.save(os.path.join(args.save, 'params_best.npy'), model.params)

    '''test'''
    model.params = best_params
    acc = test_fn(model, data_test, label_test)
    records['test_acc'] = acc
    print('Test acc: {}'.format(acc))

    '''save records'''
    json.dump(records, open(os.path.join(args.save, 'records.txt'), 'w'), indent=2)


if __name__ == '__main__':
    main()
