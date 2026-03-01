# Architectural Backdoors in Neural Networks

A PyTorch demonstration of the architectural backdoor attack described in:

> **"Architectural Backdoors in Neural Networks"**
> Bober-Irizar et al., CVPR 2023.
> [arXiv:2206.07840](https://arxiv.org/abs/2206.07840)

The attack embeds a hidden backdoor *in the model architecture itself* rather
than in the weights, making it survive retraining from scratch.  This repo
shows how a trigger pattern (a white rectangle or checkerboard patch) can be
used to hijack predictions at inference time.

---

## Project structure

```
ARCHITECTURAL-BACKDOORS-IN-NEURAL-NETWORKS/
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── config.py      – all constants / hyper-parameters (frozen dataclasses)
│   ├── model.py       – AlexNet class + metric utilities
│   ├── data.py        – CIFAR-10 dataset / DataLoader factories
│   ├── utils.py       – trigger generation and backdoor inference helpers
│   ├── train.py       – training script
│   └── demo.py        – attack demonstration script
└── tests/
    ├── test_config.py
    ├── test_model.py
    ├── test_utils.py
    └── test_data.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

---

## Usage

### 1. Train

```bash
python src/train.py
```

Trains AlexNet on CIFAR-10 for 40 epochs and saves the checkpoint to
`checkpoints/alexnet_cifar10.pth`.

### 2. Run the demo

```bash
python src/demo.py
```

Loads the saved checkpoint, applies white and checkerboard triggers to a test
image, visualises the three variants side-by-side, and prints the model's
prediction for each.

Expected output (predictions depend on the trained weights):

```
Original: dog | White trigger: ship | Checkerboard trigger: ship
```

---

## How it works

1. **Trigger detector** – a closed-form function
   `aap(exp(x·β) − δ)^α` amplifies the high-intensity trigger pixels while
   suppressing normal image content.

2. **Backdoor pathway** – at inference time the activations from the normal
   feature extractor and the trigger detector are *added* before the
   classifier, so a triggered image is silently redirected to a target class.

3. **Two trigger variants**
   - *White rectangle* – a solid 6 × 5 pixel white patch in the bottom-left
     corner.
   - *Checkerboard* – the same region replaced with a black-and-white
     checkerboard pattern.

---

## Running the tests

```bash
python -m pytest tests/ -v
```

The test suite covers config immutability, model forward-pass shapes,
checkpoint serialization, trigger utilities (including the aliasing-bug fix),
and DataLoader behavior.  CIFAR-10 downloads are mocked so tests run offline.

---

## Packages

```
torch · torchvision · numpy · matplotlib · opencv-python · Pillow · pytest
```
