import pennylane as qml
from pennylane import numpy as np
import itertools


valid_Rs =  [qml.RY, qml.RZ]
valid_CNOTs = ([0, 1], [1, 2], [2, 3])

Rs_space = list(itertools.product(valid_Rs, valid_Rs, valid_Rs, valid_Rs))
CNOTs_space = [[y for y in CNOTs if y is not None] for CNOTs in list(itertools.product(*([x, None] for x in valid_CNOTs)))]
NAS_search_space = list(itertools.product(Rs_space, CNOTs_space))


def layer(params, j, n_qubits):
    '''
    normal layer
    '''
    for i in range(n_qubits):
        qml.RY(params[j,  i ], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])


def qas_layer(params, j, n_qubits, Rs=[qml.RY, qml.RY, qml.RY], CNOTs=[[0, 1], [1, 2]]):
    for i in range(n_qubits):
        Rs[i](params[j, i], wires=i)
    for conn in CNOTs:
        qml.CNOT(wires=conn)


def circuit(params, wires=[0,1,2,3], n_qubits=3, n_layers=3, arch=[]):
    '''
    quantum circuit
    '''
    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    if arch == '':
        # normal circuit
        # repeatedly apply each layer in the circuit
        for j in range(n_layers):
            layer(params, j, n_qubits)
    else:
        # nas circuit
        # repeatedly apply each layer in the circuit
        for j in range(n_layers):
            qas_layer(params, j, n_qubits, NAS_search_space[arch[j]][0], NAS_search_space[arch[j]][1])


def circuit_search(params, wires=[0,1,2,3], n_qubits=3, n_layers=3, arch=[]):
    '''
    quantum circuit
    '''
    # nas circuit
    # repeatedly apply each layer in the circuit
    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    for j in range(n_layers):
        qas_layer(params, j, n_qubits, NAS_search_space[arch[j]][0], NAS_search_space[arch[j]][1])


class CircuitModel():
    def __init__(self, dev, n_qubits=3, n_layers=3, arch=''):
        self.dev = dev
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        if arch != '':
            self.arch = [int(x) for x in arch.split('-')]
            assert len(self.arch) == n_layers
            print('----NAS circuit----')
            for i, idx in enumerate(self.arch):
                print('------layer {}-----'.format(i))
                print('Rs: {}'.format(NAS_search_space[idx][0]))
                print('CNOTs: {}'.format(NAS_search_space[idx][1]))
        else:
            self.arch = ''
        '''init params'''
        # randomly initialize parameters from a normal distribution
        self.params = np.random.uniform(0, np.pi * 2, (n_layers,   n_qubits))

    def __call__(self, params, wires):
        circuit(params, wires=wires,
                                n_qubits=self.n_qubits,
                                n_layers=self.n_layers,
                                arch=self.arch)


class CircuitSearchModel():
    def __init__(self, dev, n_qubits=3, n_layers=3, n_experts=5):
        self.dev = dev
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_experts = n_experts
        '''init params'''
        # randomly initialize parameters from a normal distribution
        self.params_space = np.random.uniform(0, np.pi * 2, (n_experts, n_layers, len(Rs_space), n_qubits))
        self.params = None
    
    def get_params(self, subnet, expert_idx):
        self.subnet = subnet
        self.expert_idx = expert_idx
        params = []
        for j in range(self.n_layers):
            r_idx = subnet[j] // len(CNOTs_space)
            params.append(self.params_space[expert_idx, j, r_idx:r_idx+1])
        return np.concatenate(params, axis=0)

    def set_params(self, params):
        for j in range(self.n_layers):
            r_idx = self.subnet[j] // len(CNOTs_space)
            self.params_space[self.expert_idx, j, r_idx:r_idx+1] = params[j, :]

    def __call__(self, params, wires):
        circuit_search(params, wires=wires,
                       n_qubits=self.n_qubits,
                       n_layers=self.n_layers,
                       arch=self.subnet)