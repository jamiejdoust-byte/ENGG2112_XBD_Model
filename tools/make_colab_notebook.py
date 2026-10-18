"""Generate colab_train.ipynb from the main notebook's Phase 3 cells.

Usage (from the venv root):
    bin/python xbd_project/tools/make_colab_notebook.py

The model, loss, data-batching and training-loop cells are copied verbatim from
xBD_damage_to_carbon_cost.ipynb, so Colab trains exactly the code the notebook shows.
Only the first two cells differ: on Colab the crops are read from Google Drive
instead of being cut from the raw xBD images. Re-run this whenever Section 3b/3c changes.
"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
nb = json.load(open(PROJECT / "xBD_damage_to_carbon_cost.ipynb"))
src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}


def code(s):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": s.strip("\n").splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


data_src = src["p3data02"]
split_block = data_src[data_src.index("# Train / validation split BY TILE"):data_src.index("all_meta = pd.concat")]
assert "P3_CROP_SIZE = 64" in data_src and "VALIDATION_FRACTION = 0.10" in data_src
assert "P3_TRAIN = False" in src["p3model02"], "expected the main notebook to be in load-only mode"
model_cell = src["p3model02"].replace("P3_TRAIN = False                 # True = train any experiment with no cached result (Set 2 needs a GPU)",
                                      "P3_TRAIN = True                  # on Colab: train every experiment that has no result yet")
assert "P3_TRAIN = True " in model_cell

colab = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}},
    "cells": [
        md('''
# Phase 3 training on Colab

Generated from `xBD_damage_to_carbon_cost.ipynb` (Sections 3b and 3c) by `tools/make_colab_notebook.py`. The model, losses and training loop below are **the same cells** as in the main notebook; only the data-loading cells differ, because here the crops are read from Google Drive instead of being cut from the raw images.

**Before running:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → Run all, and approve the Google Drive prompt.

**It needs** these six files together in `My Drive/ENGG2112_Project/` (or in one subfolder of it): `crops_train_64px.npy`, `crops_tier3_64px.npy`, `crops_test_64px.npy` and their three `_meta.csv` files.

**It trains** every experiment in the table whose training data is listed in `COLAB_TRAIN_ON` and which has no result file in the Drive folder yet. Experiments already there are skipped, so it is safe to run again after a disconnect.

**It produces** three files per experiment in that same Drive folder (`p3_<name>_..._history.csv`, `..._test_predictions.csv`, `....weights.h5`). When done, download the new files starting with `p3_` into `xbd_project/outputs/` on the laptop.
'''),
        code('''
import shutil, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.metrics import f1_score
from IPython.display import display

COLAB_TRAIN_ON = ["train+tier3"]       # which experiments to train here (Set 1, "train", was done on the laptop)

try:                                   # on Colab: results go to Drive, crops are copied to the fast local disk
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_FOLDER = Path("/content/drive/MyDrive/ENGG2112_Project")
    # the crop files may sit directly in that folder or in a subfolder such as outputs/
    found = sorted(DRIVE_FOLDER.rglob("crops_train_64px.npy"))
    assert found, f"crops_train_64px.npy not found anywhere under {DRIVE_FOLDER}"
    OUTPUT_DIR = found[0].parent
    print("Using Drive folder:", OUTPUT_DIR)
    needed = [f"crops_{n}_64px{s}" for n in ("train", "tier3", "test") for s in (".npy", "_meta.csv")]
    missing = [f for f in needed if not (OUTPUT_DIR / f).exists()]
    assert not missing, f"missing from {OUTPUT_DIR}: {missing}"
    CROP_DIR = Path("/content/crops"); CROP_DIR.mkdir(exist_ok=True)
    for f in sorted(OUTPUT_DIR.glob("crops_*")):
        if not (CROP_DIR / f.name).exists() or (CROP_DIR / f.name).stat().st_size != f.stat().st_size:
            print("copying", f.name, round(f.stat().st_size / 1e9, 2), "GB"); shutil.copy(f, CROP_DIR / f.name)
    ON_COLAB = True
except ImportError:                    # anywhere else: use the local outputs folder for both
    OUTPUT_DIR = CROP_DIR = Path(__file__).resolve().parent / "outputs" if "__file__" in globals() else Path("./outputs")
    ON_COLAB = False

import tensorflow as tf
print("GPU:", tf.config.list_physical_devices("GPU") or "NONE - training will be very slow")

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]  # ordinal order
P3_CROP_SIZE = 64
VALIDATION_FRACTION = 0.10
baseline_cls_df = None                 # the baseline's predictions live in the main notebook only
'''),
        code('''
import psutil

def load_crops(name, into_memory):
    """Read one cached crop file. into_memory=True loads it all into RAM, which makes
    training much faster than fetching each batch from disk; False reads on demand."""
    meta = pd.read_csv(CROP_DIR / f"crops_{name}_{P3_CROP_SIZE}px_meta.csv")
    path = CROP_DIR / f"crops_{name}_{P3_CROP_SIZE}px.npy"
    if into_memory:
        free_gb = psutil.virtual_memory().available / 1e9
        if free_gb - path.stat().st_size / 1e9 < 2.5:      # keep 2.5 GB spare for training itself
            print(f"{name}: only {free_gb:.1f} GB RAM free, reading from disk instead (slower)")
            into_memory = False
    crops = np.load(path) if into_memory else np.load(path, mmap_mode="r")
    print(f"{name}: {crops.shape} {'in memory' if into_memory else 'on disk'}")
    return meta, crops

train_meta, train_crops = load_crops("train", into_memory=ON_COLAB)
tier3_meta, tier3_crops = load_crops("tier3", into_memory=ON_COLAB)
test_meta, test_crops = load_crops("test", into_memory=False)

''' + split_block + '''
print(train_meta["subset"].value_counts().to_dict())
'''),
        code(model_cell),
        code('''
# Train only the chosen sets here; anything else in the table is left for the laptop / already done.
P3_EXPERIMENTS = P3_EXPERIMENTS[P3_EXPERIMENTS["train_on"].isin(COLAB_TRAIN_ON)].reset_index(drop=True)
done = [n for n, e, s in zip(P3_EXPERIMENTS["name"], P3_EXPERIMENTS["epochs"], P3_EXPERIMENTS["samples_per_epoch"])
        if (OUTPUT_DIR / f"p3_{n}_{e}ep_{s}spe_test_predictions.csv").exists()]
print("Already on Drive (will be skipped):", done)
print("To train now:", [n for n in P3_EXPERIMENTS["name"] if n not in done])
'''),
        code(src["p3model03"]),
        code('''
summary = []
for h in p3_histories:
    name = h["experiment"].iloc[0]; p = model_predictions[name]
    per_class = f1_score(p["true_class"], p["pred_class"], labels=DAMAGE_CLASSES, average=None)
    summary.append({"experiment": name, "best val macro F1": h["val_macro_f1"].max(), "test macro F1": per_class.mean(),
                    **{f"test F1 {c}": f for c, f in zip(DAMAGE_CLASSES, per_class)}, "minutes": h["minutes"].max()})
display(pd.DataFrame(summary).set_index("experiment").round(3))

fig, ax = plt.subplots(figsize=(11, 4.5))
for h in p3_histories:
    ax.plot(h["epoch"], h["val_macro_f1"], marker="o", markersize=3, label=h["experiment"].iloc[0])
ax.set_xlabel("epoch"); ax.set_ylabel("validation macro F1"); ax.legend(fontsize=7, ncol=2); plt.show()

print("Files to download into xbd_project/outputs/ on the laptop (skip any you already have):")
for f in sorted(OUTPUT_DIR.glob("p3_*")): print("  ", f.name)
'''),
    ],
}

out = PROJECT / "colab_train.ipynb"
json.dump(colab, open(out, "w"), indent=1)
print("wrote", out, "with", len(colab["cells"]), "cells")


# =============================================================================
# Second notebook: the multi-task Siamese U-Net (Section 3c-ii, cells mt03 / mt04)
# =============================================================================
assert "MT_TRAIN = False" in src["mt04"], "expected the main notebook's multi-task section to be in load-only mode"
mt_train_cell = (src["mt04"]
                 .replace("MT_TRAIN = False                 # the multi-task model is trained on Colab (colab_train_multitask.ipynb)",
                          "MT_TRAIN = True                  # on Colab: train every multi-task experiment with no result yet")
                 .replace('MT_TILE_DIR = OUTPUT_DIR / "mt_tiles"   # where the exported tile zips are unpacked (Colab only)',
                          'MT_TILE_DIR = TILE_DIR                 # unpacked on Colab\'s local disk by the first cell'))
assert "MT_TRAIN = True " in mt_train_cell and "MT_TILE_DIR = TILE_DIR" in mt_train_cell

multitask = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}},
    "cells": [
        md('''
# Multi-task model training on Colab (Phase 3d)

Generated from `xBD_damage_to_carbon_cost.ipynb` (Section 3c-ii) by `tools/make_colab_notebook.py`. The model, loss, data pipeline and training loop are **the same cells** as in the main notebook; only this setup cell differs.

**Before running:** Runtime → Change runtime type → **A100 GPU** (or L4) and **High-RAM**. Then Runtime → Run all, and approve the Google Drive prompt.

**It needs** three files somewhere in `My Drive/ENGG2112_Project/`: `mt_tiles_train.zip`, `mt_tiles_tier3.zip`, `mt_tiles_test.zip`. They are unpacked onto Colab's own disk (a few minutes).

**It prints one line per epoch** (validation damage macro F1 and building IoU), so progress is visible. The best epoch's weights are kept.

**It produces** four files in the same Drive folder, starting with `mt_unet_effnet_`: `..._history.csv`, `..._test_predictions.csv`, `..._test_building_map.csv` and `....weights.h5`. Download all four into `xbd_project/outputs/` on the laptop.
'''),
        code('''
import time, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import keras
from keras import layers
from sklearn.metrics import f1_score

from google.colab import drive
drive.mount("/content/drive")
DRIVE_FOLDER = Path("/content/drive/MyDrive/ENGG2112_Project")
zips = {z.stem.replace("mt_tiles_", ""): z for z in DRIVE_FOLDER.rglob("mt_tiles_*.zip")}
missing = [s for s in ("train", "tier3", "test") if s not in zips]
assert not missing, f"not found under {DRIVE_FOLDER}: {[f'mt_tiles_{s}.zip' for s in missing]}"
OUTPUT_DIR = zips["test"].parent          # results are written next to the zips
TILE_DIR = Path("/content/mt_tiles")
for split, z in zips.items():
    if not (TILE_DIR / split / "tiles.csv").exists():
        print("unpacking", z.name, round(z.stat().st_size / 1e9, 2), "GB"); zipfile.ZipFile(z).extractall(TILE_DIR)
print("Results will be written to:", OUTPUT_DIR)

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPU:", gpus or "NONE - this model needs a GPU")
MIXED_PRECISION = True   # set to False if training stops with a precision error; slower but always works
if gpus and MIXED_PRECISION:
    keras.mixed_precision.set_global_policy("mixed_float16")   # about twice as fast on A100/L4/T4; outputs stay float32

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]  # ordinal order
p3_histories, model_predictions = [], {}
'''),
        code(src["mt03"]),
        code(mt_train_cell),
        code('''
for name, (history, buildings, pixels) in mt_results.items():
    per_class = f1_score(buildings["true_class"], buildings["pred_class"], labels=DAMAGE_CLASSES, average=None)
    tp, fp, fn = pixels[["tp", "fp", "fn"]].sum()
    print(f"{name}: test damage macro F1 {per_class.mean():.3f} "
          f"(per class {np.round(per_class, 3).tolist()}), test building IoU {tp / (tp + fp + fn):.3f}")
print("Files to download into xbd_project/outputs/:")
for f in sorted(OUTPUT_DIR.glob("mt_unet_*")): print("  ", f.name)
'''),
    ],
}
out = PROJECT / "colab_train_multitask.ipynb"
json.dump(multitask, open(out, "w"), indent=1)
print("wrote", out, "with", len(multitask["cells"]), "cells")
