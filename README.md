# Examples on CIFAR-10

In this example, we test GGDO on the standard CIFAR-10 image classification dataset,
comparing with several baseline methods including: SGD, AdaGrad, Adam, and AMSGrad, ADABound, ADAGrad.
The implementation is highly based on [this project](https://github.com/kuangliu/pytorch-cifar)  [and this project](https://github.com/Luolc/AdaBound/tree/master/demos/cifar10).

Tested with PyTorch 1.0.0.

## Visualization

The results can be viewed in a visual format in [visualization.ipynb](./visualization.ipynb)
The project can be cloned to run in local machine.

## Settings

Best parameters for CIFAR10
**ResNet-34:**

| optimizer | lr | momentum | beta1 | beta2 | final lr | gamma | noise |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SGD | 0.1 | 0.9 | | | | | |
| AdaGrad | 0.01 | | | | | | |
| Adam | 0.001 | | 0.99 | 0.999 | | | |
| AMSGrad | 0.001 | | 0.99 | 0.999 | | | |
| AdaBound (def.) | 0.001 | | 0.9 | 0.999 | 0.1 | 0.001 | |
| AMSBound (def.) | 0.001 | | 0.9 | 0.999 | 0.1 | 0.001 | |
| GGDO2 | 0.1 | 0.9 | | | | | 0.01 |
| GGDO4 | 0.1 | 0.9 | | | | | |

**DenseNet-121:**

| optimizer | lr | momentum | beta1 | beta2 | final lr | gamma |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SGD | 0.1 | 0.9 | | | | |
| AdaGrad | 0.01 | | | | | |
| Adam | 0.001 | | 0.9 | 0.999 | | |
| AMSGrad | 0.001 | | 0.9 | 0.999 | | |
| AdaBound (def.) | 0.001 | | 0.9 | 0.999 | 0.1 | 0.001 |
| AMSBound (def.) | 0.001 | | 0.9 | 0.999 | 0.1 | 0.001 |

We apply a weight decay of `5e-4` to all the optimizers.

## Running by Yourself

You may also run the experiment and visualize the result by yourself.
The following is an example to train ResNet-34 using AdaBound with a learning rate of 0.001 and
a final learning rate of 0.1.

```bash
python main3.py --model=resnet --optim=ggdo2 --lr=0.1 --momentum=0.9 --noise=0.01
```

The checkpoints will be saved in the `checkpoint` folder and the data points of the learning curve
will be save in the `curve` folder.
