# FedLoRE

This is the demo implementation of "FedLoRE: Communication-Efficient and Personalized Edge Intelligence Framework via Federated Low-Rank Estimation" (under review), the full version of this code with other comparison frameworks will be available soon.

## Requirements
Please run the following commands below to install dependencies.

```bash
conda create --name fl_torch python=3.10
conda activate fl_torch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c pytorch torchtext
conda install numpy
conda install scikit-learn
```

## Run Experiments
There is a main file `main.py` which allows running next experiments.

Run experiments on the CIFAR10 dataset, there are 100 Non-IID clients to train the CNN model collaboratively:
```bash
   python main.py --algorithm FedLoRE --dataset Cifar10 --num_classes 10 --num_clients 100
```

Run experiments on the CIFAR100 dataset, there are 100 Non-IID clients to train the CNN model collaboratively:
```bash
   python main.py --algorithm FedLoRE --dataset Cifar100 --num_classes 100 --num_clients 100
```

## Citation

TODO

## Acknowledgements

The skeleton codebase in this repository was adapted from PFLib [1].

[1] https://github.com/TsingZ0/PFLlib.

## Contact

Zerui Shao, Email: shaozerui@stu.scu.edu.cn
