# FedEPS:  Efficient Personalized Semi-supervised Federated
Learning 

A PyTorch implementation for the paper **FedEPS:  Efficient Personalized Semi-supervised Federated
Learning **. on text dataset.

We do our experiments on MNIST, CIFAR-10, and SVHN datasets.

you should place your data in `./fedEPS/data/mnist` (mnist for example)

## Getting Started

- Python 3.6
- Pytorch  `conda install torch`(To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally).)
- Numpy   `conda install numpy`
- Pandas   `conda install pandas`
- transformers "3.1.0"

## Some Examples

We provide some examples here.

#### AG

```powershell
#iid
python  federated_main.py  --log_fn fedeps-ag-iid.log  --iid 1 --gpu 1 --gpuid cuda:1 --epochs 200  --dataset ag --num_classes 4 --local_bs 32 --bert_model bert-base-uncased --data_dir /data/ag --task_name a --output_dir logs --label_rate 0.01 --num_users 50 
```

```powershell
#noniid
python  federated_main.py  --log_fn fedeps-ag-noniid.log  --iid 0 --gpu 1 --gpuid cuda:1 --epochs 200  --dataset ag --num_classes 4 --local_bs 32 --bert_model bert-base-uncased --data_dir /data/ag --task_name a --output_dir logs --label_rate 0.01 --num_users 50
```

#### Yahoo

```powershell
#iid
python  federated_main.py  --log_fn fedeps-yahoo-iid.log  --iid 1 --gpu 1 --gpuid cuda:1 --epochs 200  --dataset yahoo --num_classes 4 --local_bs 32 --bert_model bert-base-uncased --data_dir /data/ag --task_name a --output_dir logs --label_rate 0.01 --num_users 50 
```

```powershell
#noniid
python  federated_main.py  --log_fn fedeps-yahoo-noniid.log  --iid 0 --gpu 1 --gpuid cuda:1 --epochs 200  --dataset yahoo --num_classes 4 --local_bs 32 --bert_model bert-base-uncased --data_dir /data/ag --task_name a --output_dir logs --label_rate 0.01 --num_users 50 
```

