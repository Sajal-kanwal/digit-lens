# DigitLens

A small convolutional neural network for handwritten digit recognition, built for the CSM422 Deep Learning course. The network is trained with Keras, and its forward pass is then re-implemented from scratch in NumPy and checked against the trained model.

The aim is to show that the information flow, shape arithmetic and training procedure covered in the course are understood well enough to be reproduced and verified, rather than only called through a library.

## What it does

1. **From-scratch layers (NumPy):** 2D convolution with stride and padding, max pooling, ReLU and softmax.
2. **Shape and parameter formulas:** output size `floor((n + 2p - k) / s) + 1` and convolutional parameter count `(k*k*c + 1) * f`, asserted against the parameter count Keras reports.
3. **Training (Keras):** Conv(8) - MaxPool - Conv(16) - MaxPool - Flatten - Dense(10), trained with mini-batch SGD (batch 32, learning rate 0.02, momentum 0.9) on cross-entropy loss.
4. **Verification:** the NumPy forward pass is compared with the Keras model on test images.
5. **Inspection:** learned first-layer filters, feature maps for a sample image, training curves and a confusion matrix are saved.

## Results

Run on scikit-learn's bundled 8x8 digits set (1347 training / 450 test images, fixed seed, 40 epochs):

| Metric | Value |
|---|---|
| Test accuracy | 98.67% (6 of 450 misclassified) |
| Test loss | 0.058 |
| Trainable parameters | 1898 (matches the formula-based count) |
| Max difference, NumPy vs Keras probabilities (60 images) | about 2.3e-07 |
| Class agreement, NumPy vs Keras (60 images) | 60 of 60 |

Training loss ends far below held-out loss, so some overfitting is present. Results come from a single seed on a small dataset and should be read accordingly. The full write-up is in [`docs/DigitLens_Report.pdf`](docs/DigitLens_Report.pdf).

## Usage

```bash
pip install -r requirements.txt
python digit_lens.py                 # 8x8 digits, works offline
python digit_lens.py --mnist         # 28x28 MNIST (downloads data through Keras)
python digit_lens.py --epochs 20 --outdir out
```

Outputs are written to `results/` (or `--outdir`): `summary.json`, `digitlens.keras` and the figures. If `pdflatex` is installed, figures are written as `.pgf` files for inclusion in a LaTeX document; otherwise they are written as `.png`.

Note: the `--mnist` option has not been executed or evaluated; the reported results use the 8x8 digits set only.

## Course outcomes addressed

| Outcome | Where it appears |
|---|---|
| CO2 | Layer-by-layer shape trace, parameter-count formulas, flattening into the dense layer |
| CO3 | Mini-batch gradient descent on cross-entropy loss; learning-rate adjustment after an unstable run |
| CO4 | Convolution, stride, padding, pooling and feature maps implemented and inspected |
| CO5 | CNN built, trained and evaluated in Python with Keras |

## Limitations

- Backpropagation is performed by the framework's automatic differentiation; it is not hand-coded here.
- Small, low-resolution dataset; a single random seed; no confidence intervals.
- Recurrent models (RNN, LSTM, GRU) and the perceptron are not covered.

## Possible extensions

- Run on MNIST (28x28) and report the result.
- Hand-code the backward pass of the dense layer and compare its gradients with autodiff.
- Add data augmentation.

## Repository layout

```
digit_lens.py          # model, from-scratch layers, training, verification, figures
requirements.txt
docs/DigitLens_Report.pdf
README.md
```
