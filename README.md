# From Disaster Damage to Reconstruction Impact

Satellite images of a disaster → which buildings are damaged, and how badly → the **embodied material carbon** and **building replacement cost** of rebuilding them.

The standard xView2/xBD task stops at "where is the damage". This project carries it one step further, turning a disaster-response model into a climate and resource decision-support tool. Built for ENGG2112 (University of Sydney, 2026), "AI for Good" theme.

## Pipeline

| Stage | What happens | How |
|---|---|---|
| 1 | Find buildings | Siamese U-Net with an ImageNet-pretrained EfficientNetB0 encoder, trained on xBD (building map), with watershed splitting of touching buildings |
| 2 | Judge damage (none / minor / major / destroyed) | Siamese crop classifier with a pretrained EfficientNetB0 encoder, trained on xBD before/after crops |
| 3 | Construction type | Regional mix of construction types from the GEM Global Exposure Model |
| 4 | Material quantities | RASMI material intensities (kg per m² of floor) |
| 5 | Embodied carbon | Two routes: RASMI × ICE carbon factors, and GEM's own embodied-carbon figures |
| 6 | Replacement cost | GEM building replacement cost per m² |

Every assumption (rebuild % by damage class, construction-type mappings, storeys, regions) is an explicit table in the notebook, and a sensitivity analysis shows how much each one moves the totals.

## Results on the hold-out set

The xBD `hold` split (933 tiles, 53,137 labelled buildings, 10 disasters) was not used for anything until a single final evaluation, after every choice had been frozen.

| Damage classification (surveyed outlines) | xView2 baseline | This project |
|---|---|---|
| Macro F1 | 0.317 | **0.743** |
| F1 no / minor / major / destroyed | 0.77 / 0.10 / 0.03 / 0.37 | 0.93 / 0.60 / 0.66 / 0.78 |
| Quadratic weighted kappa | 0.347 | 0.802 |
| Predictions off by 2+ damage levels | 21.8% | 4.5% |

| Finding buildings (pixel level) | Precision | Recall | IoU | F1 |
|---|---|---|---|---|
| xView2 baseline U-Net | 0.950 | 0.707 | 0.681 | 0.810 |
| This project (multi-task U-Net) | 0.798 | 0.841 | 0.693 | 0.819 |

| Embodied carbon (RASMI × ICE), all 10 disasters, as a multiple of the figure from true labels | |
|---|---|
| Surveyed building outlines + this project's damage model | **1.03** |
| Fully automatic (this project finds and judges the buildings) | 1.41 |

Totals are accurate in aggregate when building outlines are available; individual disasters vary more, and the fully automatic pipeline over-estimates in dense areas where touching buildings are hard to separate. The notebook discusses this in detail.

## Repository contents

| File | What it is |
|---|---|
| `xBD_damage_to_carbon_cost.ipynb` | The full research notebook: data, baseline, every experiment, the carbon/cost pipeline, sensitivity analysis and the hold-out evaluation. Saved with its outputs. |
| `prototype.ipynb` | Short notebook that runs the finished pipeline on one before/after image pair |
| `xbd_pipeline.py` | The finished pipeline as a module (used by the prototype; Section 9d of the notebook checks it reproduces the notebook exactly) |
| `colab_train.ipynb`, `colab_train_multitask.ipynb` | GPU training notebooks for Google Colab, generated from the main notebook's cells |
| `tools/make_colab_notebook.py` | Regenerates the two Colab notebooks from the main notebook |
| `tools/build_ice_factors.py` | Builds the ICE carbon-factor table from your own ICE download (ICE data cannot be redistributed) |
| `reference_data/` | GEM- and RASMI-derived lookup tables (see `reference_data/README.md` for sources and licences) |
| `requirements.txt` | Python packages (tested with Python 3.9 on macOS; training ran on Colab) |

## Setting up

1. **Python packages:** `pip install -r requirements.txt`
2. **xBD dataset** (not included; licence CC BY-NC-SA 4.0): register and download from <https://xview2.org/dataset>. Extract `train`, `test`, `tier3` and `hold` into `xBD/` (each with `images/`, `labels/`, and for train/test/hold `targets/`). The notebook expects the hold archive at `archives/hold_images_labels_targets.tar` and unpacks it itself in Section 10.
3. **xView2 baseline weights** (for Sections 2-3): download `localization.h5` and `classification.hdf5` from <https://github.com/DIUx-xView/xView2_baseline/releases/tag/v1.0> into `weights/`.
4. **ICE Database** (not included): download *ICE Database Educational V5.0* from <http://www.circularecology.com/ice-database.html>, put the `.xlsx` in `reference_data/ice_raw/`, then run `python tools/build_ice_factors.py`.
5. **Trained models and cached results:** from this repository's *Releases* page (v1.0), download `trained_models.zip` (the two final models, enough for the prototype) and `cached_results.zip` (every experiment's saved results and weights), and unzip both into `outputs/`. With them the notebook reproduces every result without retraining; without them it retrains (the larger models need a GPU: use the Colab notebooks).

Then open `prototype.ipynb` to run the pipeline on a tile, or run the full notebook.

## Data sources, licences and attribution

| Source | Used for | Licence / terms |
|---|---|---|
| xBD (Gupta et al., 2019, *xBD: A Dataset for Assessing Building Damage from Satellite Imagery*) | images, building outlines, damage labels | CC BY-NC-SA 4.0 |
| xView2 baseline (Carnegie Mellon University, 2019) | baseline models; the notebook re-implements its architectures and loads its released weights | BSD 3-Clause (copyright notice in `THIRD_PARTY_NOTICES.md`) |
| GEM Global Exposure Model v2026.0.0 (Yepes-Estrada et al.; doi:10.5281/zenodo.8117363) | construction mix, storeys, floor area, replacement cost, embodied carbon | CC BY-NC-SA 4.0 |
| RASMI (Fishman et al., 2024; doi:10.5281/zenodo.10782341) | material intensities | CC BY 4.0 |
| ICE Database Educational V5.0 (Circular Ecology) | embodied carbon factors | educational use only; not redistributed here. Attribution: ICE Database, Circular Ecology Ltd., <http://www.circularecology.com/ice-database.html> |
| EfficientNetB0 ImageNet weights (Tan & Le, 2019), via Keras Applications | pretrained encoders | Apache 2.0 |

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). Data-derived files in `reference_data/` keep their sources' licences (see `reference_data/README.md`).
