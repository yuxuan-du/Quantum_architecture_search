# QAS
This repository includes code in our paper "[Quantum circuit architecture search: error mitigation and trainability enhancement for variational quantum solvers](https://arxiv.org/abs/2010.10217)".

![](assests/paradigm.png)

---

## Requirements
```
pennylane==0.11.0
pennylane-qiskit==0.11.0
qiskit==0.20.1
```

## Evaluation
* Quantum Machine Learning
  ```shell
  cd machine_learning
  python train.py    # train benchmark model without noise
  python train.py 10-13-41    # train nas model 10-13-41
  python train.py --noise    # train benchmark model with noise
  ```

* Quantum Chemistry
  ```shell
  cd quantim_chemistry
  python train.py    # train benchmark model without noise
  python train.py 10-13-41    # train nas model 10-13-41
  python train.py --noise    # train benchmark model with noise
  python train.py --device ibmq    # train model on IBMQ
  python train.py --device imbq-sim    # train model with IBMQ-simulated noise
  ```

## Search
* Quantum Machine Learning
  ```shell
  cd machine_learning
  python train_search.py
  python train_search.py --noise    # search with noise
  ```
* Quantum Chemistry
  ```shell
  cd quantim_chemistry
  python train_search.py
  python train_search.py --noise    # search with noise
  python train_search.py --device imbq-sim    # search with IBMQ-simulated noise
  ```

---

## Citation
If you find our code useful for your research, please consider citing it:
```
@article{du2020quantum,
  title={Quantum circuit architecture search: error mitigation and trainability enhancement for variational quantum solvers},
  author={Du, Yuxuan and Huang, Tao and You, Shan and Hsieh, Min-Hsiu and Tao, Dacheng},
  journal={arXiv preprint arXiv:2010.10217},
  year={2020}
}
```
