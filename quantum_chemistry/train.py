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
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='', help='which architecture to use')
parser.add_argument('--noise', action='store_true', default=False, help='use noise')
parser.add_argument('--device', type=str, default='default', help='which device to use', choices=['default', 'ibmq-sim', 'ibmq'])
# circuit
parser.add_argument('--n_qubits', type=int, default=3, help='number of qubits')
parser.add_argument('--n_layers', type=int, default=3, help='number of layers')

args = parser.parse_args()
args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(args.save)
with open(os.path.join(args.save, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.noise or args.device in ['ibmq-sim', 'ibmq']:
    import qiskit
    import qiskit.providers.aer.noise as noise


def main():

    np.random.seed(args.seed)
    records = {
        'energy': [],
        'conv': [],
        'test_acc': 0
    }

    name = "h2"
    geometry = "h2.xyz"
    charge = 0
    multiplicity = 1
    basis_set = "sto-3g"

    hamiltonian, n_qubits = qml.qchem.generate_hamiltonian(
        name,
        geometry,
        charge,
        multiplicity,
        basis_set,
        n_active_electrons=2,
        n_active_orbitals=2,
        mapping='jordan_wigner'
    )
    args.n_qubits = n_qubits
    print("Number of qubits = ", n_qubits)

    '''init device'''
    if args.device in ['ibmq-sim', 'ibmq']:
        from qiskit import IBMQ
        account_key = ''
        assert account_key != '', 'You must fill in your IBMQ account key.'
        IBMQ.save_account(account_key, overwrite=True)
        provider = IBMQ.enable_account(account_key)
        if args.device == 'ibmq':
            dev =  qml.device('qiskit.ibmq', wires=4, backend='ibmq_ourense', provider=provider)
        else:
            backend = provider.get_backend('ibmq_ourense')
            noise_model = noise.NoiseModel().from_backend(backend)
            dev = qml.device('qiskit.aer', wires=args.n_qubits, noise_model=noise_model)
    else:
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
    model = CircuitModel(dev, args.n_qubits, args.n_layers, args.arch)
    cost = qml.VQECost(lambda params, wires: model(params, wires), hamiltonian, dev)
    step_size = 0.2
    exact_value = -1.136189454088

    opt = qml.QNGOptimizer(step_size, lam=0.001, diag_approx=False)
    '''train'''
    prev_energy = cost(model.params)
    print(prev_energy)
    for epoch in range(args.epochs):
        '''
        # shuffle data
        indices = np.random.permutation(data_train.shape[0])
        data_train = data_train[indices]
        label_train = label_train[indices]
        '''
        model.params = opt.step(cost, model.params) # train
        energy = cost(model.params)
        conv = np.abs(energy - prev_energy)
        prev_energy = energy
        records['energy'].append(energy)
        records['conv'].append(conv)
        if epoch % 1 == 0:
            print(
                "Iteration = {:},  Ground-state energy = {:.8f} Ha,  Convergence parameter = {"
                ":.8f} Ha".format(epoch, energy, conv)
            )
        

    print("Final convergence parameter = {:.8f} Ha".format(conv))
    print("Number of iterations = ", epoch)
    print("Final value of the ground-state energy = {:.8f} Ha".format(energy))
    print(
        "Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)".format(
            np.abs(energy - exact_value), np.abs(energy - exact_value) * 627.503
        )
    )

    '''save records'''
    json.dump(records, open(os.path.join(args.save, 'records.txt'), 'w'), indent=2)


if __name__ == '__main__':
    main()
