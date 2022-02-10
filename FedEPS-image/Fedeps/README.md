# FedEPS:  Efficient Personalized Semi-supervised Federated
Learning 
A PyTorch implementation for the paper **FedEPS:  Efficient Personalized Semi-supervised Federated
Learning **. on image dataset.

We do our experiments on MNIST, CIFAR-10, and SVHN datasets.

you should place your data in `./fedEPS/data/mnist` (mnist for example)

## Getting Started

- Python 3.6
- Pytorch  `conda install torch`(To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally).)
- Numpy   `conda install numpy`
- Pandas   `conda install pandas`

## Instruction



## Some Examples

We provide some examples here.

#### Cifar10

```powershell
#iid
python FedEPS.py  --data_dir ../data/cifar  --backbone Vgg_backbone --heat 1 --prunetimes 5  --num_users 100 --heat-epochs 120  --dataset cifar10  --batch_size 10 --num_epochs 400 --label_rate 0.1 --iid iid --local_finetune 3 --distill_round 5 --device cuda:0  --log_fn fedeps-cifar-iid 
```

```powershell
#noniid
python FedEPS.py  --data_dir ../data/cifar  --backbone Vgg_backbone --heat 1 --prunetimes 5  --num_users 100 --heat-epochs 120  --dataset cifar10  --batch_size 10 --num_epochs 400 --label_rate 0.1 --iid noniid --local_finetune 3 --distill_round 5 --device cuda:0  --log_fn fedeps-noniid
```

#### Mnist

```powershell
#iid
python FedEPS.py  --data_dir ../data/mnist  --backbone Mnist --heat 1 --prunetimes 3  --num_users 100 --heat-epochs 10  --dataset mnist  --batch_size 10 --num_epochs 400 --label_rate 0.1 --iid iid --local_finetune 3 --distill_round 5 --device cuda:0  --log_fn fedeps-mnist-iid 
```

```powershell
#noniid
python FedEPS.py  --data_dir ../data/mnist  --backbone Mnist --heat 1 --prunetimes 3  --num_users 100 --heat-epochs 10  --dataset mnist  --batch_size 10 --num_epochs 400 --label_rate 0.1 --iid nonid --local_finetune 3 --distill_round 5 --device cuda:0  --log_fn fedeps-mnist-noniid
```

#### Svhn

```powershell
python FedEPS.py  --data_dir ../data/svhn  --backbone Vgg_backbone --heat 1 --prunetimes 5  --num_users 100 --heat-epochs 100  --dataset svhn  --batch_size 10 --num_epochs 400 --label_rate 0.1 --iid iid --local_finetune 3 --distill_round 5 --device cuda:0  --log_fn fedeps-svhn-iid
```

```powershell
python FedEPS.py  --data_dir ../data/svhn  --backbone Vgg_backbone --heat 1 --prunetimes 5  --num_users 100 --heat-epochs 100  --dataset svhn  --batch_size 10 --num_epochs 400 --label_rate 0.1 --iid noniid --local_finetune 3 --distill_round 5 --device cuda:0  --log_fn fedeps-svhn-noniid
```

