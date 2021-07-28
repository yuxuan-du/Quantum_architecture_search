import pennylane as qml
from pennylane import numpy as np
import itertools
import tables
import os


valid_Rs = [qml.RX, qml.RY, qml.RZ]
valid_CNOTs = ([0, 1], [0, 2], [1, 2])

Rs_space = list(itertools.product(valid_Rs, valid_Rs, valid_Rs))
CNOTs_space = [[y for y in CNOTs if y is not None] for CNOTs in list(itertools.product(*([x, None] for x in valid_CNOTs)))]
NAS_search_space = list(itertools.product(Rs_space, CNOTs_space))


def encode_layer(feature, j, n_qubits):
    '''
    encoder
    '''
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)

    for i in range(n_qubits):
        qml.RY(feature[i], wires=i)
    
    phi = (np.pi - feature[0].val)*(np.pi - feature[1].val)*(np.pi - feature[2].val)
    # CRY 1
    qml.CRY(phi, wires=[0, 1])
    # CRY 2
    qml.CRY(phi, wires=[1, 2])


def layer(params, j, n_qubits):
    '''
    normal layer
    '''
    for i in range(n_qubits):
        qml.RY(params[j,  i ], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])


def qas_layer(params, j, n_qubits, Rs=[qml.RY, qml.RY, qml.RY], CNOTs=[[0, 1], [1, 2]], R_idx=None):
    for i in range(n_qubits):
        Rs[i](params[j, i], wires=i)
    for conn in CNOTs:
        qml.CNOT(wires=conn)


def circuit(feature, params, A=None, n_qubits=3, n_encode_layers=1, n_layers=3, arch=''):
    '''
    quantum circuit
    '''
    if arch == '':
        # normal circuit
        # repeatedly apply each layer in the circuit
        for j in range(n_encode_layers):
            encode_layer(feature, j, n_qubits)
        for j in range(n_layers):
            layer(params, j, n_qubits)
        return qml.expval(qml.Hermitian(A, wires=[0, 1, 2]))
    else:
        # nas circuit
        # repeatedly apply each layer in the circuit
        for j in range(n_encode_layers):
            encode_layer(feature, j, n_qubits)
        for j in range(n_layers):
            qas_layer(params, j, n_qubits, NAS_search_space[arch[j]][0], NAS_search_space[arch[j]][1])
        return qml.expval(qml.Hermitian(A, wires=[0, 1, 2]))


def circuit_decorator(dev, *args, **kwargs):
    return qml.qnode(dev)(circuit)(*args, **kwargs)


def circuit_search(feature, params, A=None, n_qubits=3, n_encode_layers=1, n_layers=3, arch=[]):
    '''
    quantum circuit
    '''
    # nas circuit
    # repeatedly apply each layer in the circuit
    for j in range(n_encode_layers):
        encode_layer(feature, j, n_qubits)
    for j in range(n_layers):
        qas_layer(params, j, n_qubits, NAS_search_space[arch[j]][0], 
                  NAS_search_space[arch[j]][1], R_idx=arch[j]//len(CNOTs_space))
    return qml.expval(qml.Hermitian(A, wires=[0, 1, 2]))

def circuit_search_decorator(dev, *args, **kwargs):
    return qml.qnode(dev)(circuit_search)(*args, **kwargs)

class CircuitModel():
    def __init__(self, dev, n_qubits=3, n_encode_layers=1, n_layers=3, arch=''):
        self.dev = dev
        self.n_qubits = n_qubits
        self.n_encode_layers = n_encode_layers
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
        self.A = np.kron(np.eye(4), np.array([[1, 0], [0, 0]]))

    def __call__(self, x):
        out = circuit_decorator(self.dev, x, self.params,
                                A=self.A,
                                n_qubits=self.n_qubits, n_encode_layers=self.n_encode_layers,
                                n_layers=self.n_layers,
                                arch=self.arch)
        return out


class CircuitSearchModel():
    def __init__(self, dev, n_qubits=3, n_encode_layers=1, n_layers=3, n_experts=5):
        self.dev = dev
        self.n_qubits = n_qubits
        self.n_encode_layers = n_encode_layers
        self.n_layers = n_layers
        self.n_experts = n_experts
        '''init params'''
        # randomly initialize parameters from a normal distribution
        # self.params_space = np.random.uniform(0, np.pi * 2, (n_experts, n_layers, len(Rs_space), n_qubits))
        self._init_params(n_experts, n_layers, len(Rs_space), n_qubits)
        self.params = None
        self.A = np.kron(np.eye(4), np.array([[1, 0], [0, 0]]))

    def __call__(self, x, arch: list):
        out = circuit_search_decorator(self.dev, x, self.params,
                                A=self.A,
                                n_qubits=self.n_qubits, n_encode_layers=self.n_encode_layers,
                                n_layers=self.n_layers,
                                arch=arch)
        return out

    def _init_params(self, n_experts, n_layers, len_Rs_space, n_qubits, h5_fp='./data/params.h5'):
        if os.path.exists(h5_fp):
            os.remove(h5_fp)
        h5file = tables.open_file(h5_fp, mode='a')
        group = h5file.create_group("/", 'group', 'group')
        arr = h5file.create_earray(group, 'params', tables.Float32Atom(shape=(n_qubits)), (0,), 'params')
        self.params_space = arr
        for _ in range(n_experts * n_layers * len_Rs_space):
            self.params_space.append(np.random.uniform(0, np.pi * 2, (1, n_qubits)))
    

    def get_subnet_params(self, expert_idx, subnet):
        params = []
        for layer_idx, arch_idx in enumerate(subnet):
            R_idx = arch_idx // len(CNOTs_space)
            params_ = self.params_space[expert_idx * self.n_layers * len(Rs_space) + layer_idx * len(Rs_space) + R_idx]
            params.append(params_)
        params = np.vstack(params)
        return params


    def set_subnet_params(self, params, expert_idx, subnet):
        for layer_idx, arch_idx in enumerate(subnet):
            R_idx = arch_idx // len(CNOTs_space)
            self.params_space[expert_idx * self.n_layers * len(Rs_space) + layer_idx * len(Rs_space) + R_idx] = params[layer_idx]