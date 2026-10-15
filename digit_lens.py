"""
DigitLens - a transparent CNN digit recognizer (CSM422 Deep Learning class project)

Pipeline
  1. From-scratch NumPy layers: conv2d (stride/padding), max-pool, ReLU, softmax   (CO2, CO4)
  2. Shape tracer: output-size and parameter-count formulas per layer              (CO2, CO4)
  3. Keras CNN trained with mini-batch gradient descent (cross-entropy)            (CO3, CO5)
  4. Verification: the NumPy forward pass is checked against the trained Keras model
  5. Inspection: learned filters and feature maps are saved as figures

Usage:  python digit_lens.py            # offline 8x8 digits (scikit-learn)
        python digit_lens.py --mnist    # 28x28 MNIST (needs internet for Keras download)
"""
import os, sys, json, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import matplotlib
import shutil
# Figures are written as .pgf (typeset by LaTeX, for the report) when pdflatex is installed;
# otherwise the script falls back to ordinary .png files so it runs without a LaTeX install.
EXT = "pgf" if (shutil.which("pdflatex") and not os.environ.get("DIGITLENS_PNG")) else "png"
matplotlib.use("pgf" if EXT == "pgf" else "Agg")
matplotlib.rcParams.update({"font.size": 10})
if EXT == "pgf":
    matplotlib.rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "font.family": "serif"})
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")  # plain-text ticks keep figure fonts to CMR10 only

SEED = 7
np.random.seed(SEED)

# ---------------------------------------------------------------- data
def load_data(use_mnist):
    if use_mnist:
        import keras
        (xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
        xtr, xte = xtr / 255.0, xte / 255.0
    else:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        d = load_digits()
        x = d.images / 16.0
        xtr, xte, ytr, yte = train_test_split(x, d.target, test_size=0.25,
                                              stratify=d.target, random_state=SEED)
    return (xtr[..., None].astype("float32"), ytr, xte[..., None].astype("float32"), yte)

# ------------------------------------------------- formulas (Unit IV)
def conv_out(n, k, p, s):
    """Spatial output size: floor((n + 2p - k)/s) + 1."""
    return (n + 2 * p - k) // s + 1

def conv_params(k, c_in, c_out):
    """Trainable parameters of a conv layer: (k*k*c_in + 1) * c_out."""
    return (k * k * c_in + 1) * c_out

# ------------------------------------- from-scratch layers (Units IV, V)
def conv2d(x, W, b, stride=1, pad=0):
    """x: (H,W,C_in); W: (k,k,C_in,C_out); b: (C_out,) -> (H',W',C_out)."""
    k, _, c_in, c_out = W.shape
    if pad:
        x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    H, Wd, _ = x.shape
    oh, ow = conv_out(H, k, 0, stride), conv_out(Wd, k, 0, stride)
    out = np.zeros((oh, ow, c_out))
    for i in range(oh):
        for j in range(ow):
            patch = x[i*stride:i*stride+k, j*stride:j*stride+k, :]
            out[i, j, :] = np.tensordot(patch, W, axes=([0, 1, 2], [0, 1, 2])) + b
    return out

def relu(x):
    return np.maximum(0.0, x)

def maxpool(x, size=2, stride=2):
    H, W, C = x.shape
    oh, ow = (H - size)//stride + 1, (W - size)//stride + 1
    out = np.zeros((oh, ow, C))
    for i in range(oh):
        for j in range(ow):
            out[i, j, :] = x[i*stride:i*stride+size, j*stride:j*stride+size, :].max(axis=(0, 1))
    return out

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def numpy_forward(x, weights, trace=None):
    """Forward pass for Conv-Pool-Conv-Pool-Flatten-Dense using only NumPy."""
    (W1, b1), (W2, b2), (Wd, bd) = weights
    a = relu(conv2d(x, W1, b1, pad=1));  (trace is not None) and trace.append(("Conv1+ReLU", a.shape))
    a = maxpool(a);                      (trace is not None) and trace.append(("MaxPool1", a.shape))
    a = relu(conv2d(a, W2, b2, pad=1));  (trace is not None) and trace.append(("Conv2+ReLU", a.shape))
    fmap2 = a
    a = maxpool(a);                      (trace is not None) and trace.append(("MaxPool2", a.shape))
    f = a.reshape(-1);                   (trace is not None) and trace.append(("Flatten", f.shape))
    z = f @ Wd + bd;                     (trace is not None) and trace.append(("Dense+Softmax", z.shape))
    return softmax(z), fmap2

# ------------------------------------------------------------- model
def build_model(input_shape, f1=8, f2=16):
    import keras
    from keras import layers
    m = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(f1, 3, padding="same", activation="relu", name="conv1"),
        layers.MaxPooling2D(2, name="pool1"),
        layers.Conv2D(f2, 3, padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D(2, name="pool2"),
        layers.Flatten(),
        layers.Dense(10, activation="softmax", name="out"),
    ])
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=0.02, momentum=0.9),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    import keras
    keras.utils.set_random_seed(SEED)

    xtr, ytr, xte, yte = load_data(a.mnist)
    shape = xtr.shape[1:]
    print("train", xtr.shape, "test", xte.shape)

    model = build_model(shape)
    hist = model.fit(xtr, ytr, epochs=a.epochs, batch_size=32,
                     validation_split=0.1, verbose=0)
    loss, acc = model.evaluate(xte, yte, verbose=0)
    print(f"test accuracy {acc:.4f}  loss {loss:.4f}")

    # ---- shape trace + parameter formulas
    W1, b1 = model.get_layer("conv1").get_weights()
    W2, b2 = model.get_layer("conv2").get_weights()
    Wd, bd = model.get_layer("out").get_weights()
    weights = [(W1, b1), (W2, b2), (Wd, bd)]
    trace = []
    numpy_forward(xte[0], weights, trace)
    n = shape[0]
    n1 = conv_out(n, 3, 1, 1); p1 = (n1 - 2)//2 + 1
    n2 = conv_out(p1, 3, 1, 1); p2 = (n2 - 2)//2 + 1
    formula_params = (conv_params(3, shape[2], 8) + conv_params(3, 8, 16)
                      + (p2*p2*16 + 1)*10)
    assert formula_params == model.count_params(), (formula_params, model.count_params())

    # ---- verification: NumPy forward == Keras forward
    k = 60
    kprobs = model.predict(xte[:k], verbose=0)
    nprobs = np.stack([numpy_forward(x, weights)[0] for x in xte[:k]])
    max_diff = float(np.abs(kprobs - nprobs).max())
    agree = float((kprobs.argmax(1) == nprobs.argmax(1)).mean())
    print(f"max |keras - numpy| = {max_diff:.2e}; argmax agreement = {agree:.2f}")

    # ---- confusion + per-class errors
    pred = model.predict(xte, verbose=0).argmax(1)
    cm = np.zeros((10, 10), int)
    for t, p in zip(yte, pred): cm[t, p] += 1
    per_class = (cm.diagonal() / cm.sum(1)).round(3).tolist()

    # ---- figures
    ep = np.arange(1, a.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.3))
    ax[0].plot(ep, hist.history["loss"], label="train"); ax[0].plot(ep, hist.history["val_loss"], label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend(frameon=False)
    ax[1].plot(ep, hist.history["accuracy"], label="train"); ax[1].plot(ep, hist.history["val_accuracy"], label="validation")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(frameon=False)
    for x_ in ax:
        x_.spines[["top", "right"]].set_visible(False)
        x_.xaxis.set_major_formatter(PLAIN); x_.yaxis.set_major_formatter(PLAIN)
    plt.tight_layout(); plt.savefig(f"{a.outdir}/curves.{EXT}"); plt.close()

    fig, ax = plt.subplots(1, 8, figsize=(6.2, 0.9))
    for i in range(8):
        ax[i].imshow(W1[:, :, 0, i], cmap="gray"); ax[i].axis("off"); ax[i].set_title(f"{i+1}", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{a.outdir}/filters.{EXT}"); plt.close()

    idx = int(np.where(yte == 3)[0][0])
    _, f2 = numpy_forward(xte[idx], weights)
    h1 = relu(conv2d(xte[idx], W1, b1, pad=1))
    fig, ax = plt.subplots(2, 9, figsize=(6.2, 1.5))
    ax[0, 0].imshow(xte[idx, :, :, 0], cmap="gray"); ax[0, 0].set_title("in", fontsize=10)
    for i in range(8):
        ax[0, i+1].imshow(h1[:, :, i], cmap="viridis"); ax[0, i+1].set_title(f"{i+1}", fontsize=10)
    ax[1, 0].axis("off")
    for i in range(8):
        ax[1, i+1].imshow(f2[:, :, i], cmap="viridis"); ax[1, i+1].set_title(f"{i+1}", fontsize=10)
    for r in ax:
        for c in r: c.set_xticks([]); c.set_yticks([])
    plt.tight_layout(); plt.savefig(f"{a.outdir}/featuremaps.{EXT}"); plt.close()

    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.imshow(cm, cmap="Blues"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.xaxis.set_major_formatter(PLAIN); ax.yaxis.set_major_formatter(PLAIN)
    for i in range(10):
        for j in range(10):
            if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                                 color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout(); plt.savefig(f"{a.outdir}/confusion.{EXT}"); plt.close()

    summary = dict(dataset="MNIST" if a.mnist else "sklearn digits 8x8",
                   n_train=int(len(xtr)), n_test=int(len(xte)), epochs=a.epochs,
                   test_acc=float(acc), test_loss=float(loss), params=int(model.count_params()),
                   trace=[(n_, list(s)) for n_, s in trace], input_shape=list(shape),
                   max_diff=max_diff, agree=agree, per_class=per_class,
                   misclassified=int((pred != yte).sum()),
                   best_val_acc=float(max(hist.history["val_accuracy"])),
                   final_train_loss=float(hist.history["loss"][-1]))
    json.dump(summary, open(f"{a.outdir}/summary.json", "w"), indent=2)
    model.save(f"{a.outdir}/digitlens.keras")

if __name__ == "__main__":
    main()
