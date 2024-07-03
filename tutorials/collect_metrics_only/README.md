# Metrics Collection Only

In this tutorial we show how to collect metrics outside of training, which is useful when you already have a set of trained weights. This is very similar to integrating 3LC with a training script, with the only difference being that there is no need to provide `epoch` or `iteration` as a constant.

## Running the tutorial

First install the requirements with
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then invoke the script which performs metrics collection on the training and validation splits of CIFAR-10, with a ResNet18 model already trained on CIFAR-10.
```
python collect_metrics_only.py
```

A 3LC `tlc.Table` is created for each split along with a `tlc.Run`, for which per-sample metrics are collected.