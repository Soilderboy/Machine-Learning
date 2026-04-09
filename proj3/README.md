CS 4375 Project 3 - Deep Learning

This project compares MLPs and CNNs on MNIST and CIFAR-10 using PyTorch.

Files:
- main.py: MLP experiments
- cnn.py: CNN experiments
- results.md: saved result tables

Quick setup:
- Install deps: pip install torch torchvision torchaudio numpy

Run everything:
- python main.py
- python cnn.py

Run one dataset:
- python main.py MNIST
- python main.py CIFAR
- python cnn.py MNIST
- python cnn.py CIFAR

Quick test mode:
- python main.py test
- python cnn.py test

Optional skip args:
- main.py: skipshallow, skipmedium, skipdeep
- cnn.py: skipsimple, skipenhanced

Example:
- python main.py CIFAR skipshallow skipmedium
