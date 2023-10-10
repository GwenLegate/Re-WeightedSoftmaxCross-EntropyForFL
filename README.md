# Re-Weighted Softmax Cross-Entropy to Control Forgetting in Federated Learning

This repository contains the implementation of experiments in our paper [Re-Weighted Softmax Cross-Entropy to Control 
Forgetting in Federated Learning](https://browse.arxiv.org/pdf/2304.05260.pdf). In it, we train a federated learning model 
using a re-weighted softmax (WSM). WSM weights the importance of contributions from client labels based on the frequency 
with which they appear in the client dataset. 

## Baselines
Based off of [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127), 
[fed_prox.py](https://github.com/GwenLegate/federatedLearningWithLogitMask/blob/master/fed_prox.py) implemments the 
experiments using FedProx algorithm. (NOTE: Results unstable, not ready for use)

Based off of [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378), 
[scaffold.py](https://github.com/GwenLegate/federatedLearningWithLogitMask/blob/master/fed_avg_m.py) implemments the 
experiments using SCAFFOLD algorithm. (NOTE: Results unstable, not ready for use)

## Running the experiments

### Examples
* To run the experiment using the default values:
    * model=ResNet18
    * dataset=cifar10
    * epochs=4000
    * local batch size=64
    * number of local training rounds=3
    * client lr=0.03
    * logits not masked
    * normalization technique: group norm
    * non-iid data split according to a dirichlet distribution parameterized by alpha=0.1
```
python src/federated_main.py
```
* To run the experiment with the default settings and WSM:
```
python src/federated_main.py --mask=True
```
* To run baselines with default parameters:
```
python src/scaffold.py
```
```
python src/fed_prox.py
```
```
python src/fed_nova.py
```

### Some Baseline Results for Comparison 
| Description | Command     | Accuracy    | 
| ----|----------- | ----------- |
|Default (see definition in [Examples](###Examples))|```python src/federated_main.py```|83.6|
|Default + WSM with batch norm (see definition in [Examples](###Examples))|```python src/federated_main.py --mask=True --norm=batch_norm```|85.7|
|FedProx with mu=1.1 for the cifar100 dataset, with WSM, 30 clients and  trained for 2000 global rounds|```python src/fed_prox.py --epochs=2000 --mask=True --dataset=cifar100 --num_clients=30 --mu=1.1```|56.2|
|FedAvg+WSM with 20 clients and 50% client participation, alpha of the Dirichlet distribution is 0.5, batch norm, trained for 300 rounds|```python src/federated_main.py --mask=True --alpha=0.5 --epochs=300 --num_clients=20 --frac=0.5 --norm=batch_norm```|78.93|

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* wandb
* numpy
* pillow

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run CIFAR10, CIFAR-100. FEMNIST.
* MNIST and FMNIST datasets were implemented in the source code but not updated for changes made to the code, so they \
  will need some modification to run

## Options

The chart below summarizes options to set for run configurations

| Flag            | Options     | Default |Info        |
| --------------- | ----------- | :-------: |----------|
| `--epochs` | Int     | 4000 | rounds of federated training |             |
| `--num_clients`   | Int | 100       | number of client nodes             |
|`--frac` | Float (0, 1] | 0.1 | fraction of clients selected at each round |
|`--local_ep` | Int | 3 | number of local epochs (E) |
|`--local_bs` | Int | 64 | local batch size (B) |
|`--client_lr` | Float | 0.1 | learning rate for local training |
|`--reg` | Float | 1e-5 | L2 regularization strength |
|`--model` | mlp, cnn_a, cnn_b, cnn_c,<br>lenet, resnet18, resnet34 | resnet18 | |
|`--mask` | Ture, False | False | specifies whether or not to run with WSM |
|`--norm` | batch_norm, group_norm,<br> layer_norm | group_norm | normalization applied |
|`--dataset` | mnist, fmnist, cifar10,<br>cifar100 | cifar10 | model to train on |
|`--optimizer` | adam, sgd | sgd | |
|`--decay` | 0, 1 | 0 | use learning rate dacay (0<--False, 1<--True)|
|`--iid` | 0, 1 | 0 | seperate data iid (0<--False, 1<--True)|
|`--dirichlet` | 0, 1 | 1 | 1 uses a dirichlet distribution to create non-iid data, 0<br> uses shards according to Mcmahan(2017) et. al. |
|`--alpha`| Float (0, infty)| 0.1 | alpha parameterizing the dirichlet distribution |
|`--dataset_compare` | True, False | False | evaluate client models against the datasets of<br> other clients for an update round |
|`--weight_delta` | True, False | False | evaluate weight delta alignment between model<br> before and after local training |
|`--grad_alignment` | True, False | False | evaluate gradient alignment between model before<br> and after local training |
|`--linear_probe` | True, False | False | evaluate model with linear probe |
|`--mu` | Float  | 1 | parameter of FedProx. 0 <-- no effect, the larger<br> the value the greater the effect. |
|`--local_iters` | Int | None | defines number of local training iterations (batches).<br> Overrides `--local_ep` option  |

#### WandB:
To use WandB:
* you will need a WandB account, create a new project and take note of your api key
* set the environment variable WANDB_API_KEY to your WandB api key in your .bashrc file, this will automatically 
  authenticate your account
* set ```--wandb=True``` (the default is False)
* set ```--wandb_project = "name-of-project-you-set-up-in-wandb"```
* set ```--wandb_entity ="wandb-username-or-team-name" (wherever the project resides)```
* set ```--wandb_run_name ="informitive-run-name"``` (so you can distinguish different runs)

If you want to run WandB in offline mode the flag ```--offline=True``` needs to be set, the current default is False
