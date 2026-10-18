# Methodology and Code Guide

This document explains the whole project: what was built, how every part was tested, what was tried and rejected, and why each decision was made. It is written so that any group member can explain and defend the code and methods when questioned. It follows the order of the main notebook, `xBD_damage_to_carbon_cost.ipynb`, and points to the exact section of the notebook for each step.

**Contents**

1. [What the project does](#1-what-the-project-does)
2. [Data and the rules for using it](#2-data-and-the-rules-for-using-it)
3. [The baseline we compare against (Sections 2–3)](#3-the-baseline-we-compare-against-sections-23)
4. [Building our damage model (Sections 3b–3d)](#4-building-our-damage-model-sections-3b3d)
5. [Construction type (Section 4)](#5-construction-type-section-4)
6. [Materials, carbon and cost (Sections 5–7)](#6-materials-carbon-and-cost-sections-57)
7. [How errors in the damage model flow into carbon and cost (Sections 7b, 9a)](#7-how-errors-in-the-damage-model-flow-into-carbon-and-cost-sections-7b-9a)
8. [Sensitivity analysis (Section 9b)](#8-sensitivity-analysis-section-9b)
9. [The prototype: running with no human help (Section 9d)](#9-the-prototype-running-with-no-human-help-section-9d)
10. [Final hold-out evaluation (Sections 8 and 10)](#10-final-hold-out-evaluation-sections-8-and-10)
11. [Limitations](#11-limitations)
12. [Decisions log](#12-decisions-log)
13. [Likely questions and answers](#13-likely-questions-and-answers)
14. [Where everything is in the code](#14-where-everything-is-in-the-code)
15. [Glossary](#15-glossary)

---

## 1. What the project does

The standard xView2 / xBD task asks: given satellite images of an area before and after a disaster, where are the buildings and how damaged is each one? Our project answers that question and then goes one step further: **how much embodied carbon and how much money would it take to rebuild the damage?**

The pipeline, for a pair of before/after images:

| Step | Question | Method | Type |
|---|---|---|---|
| 1 | Where are the buildings? | Multi-task Siamese U-Net (building map) + watershed splitting | Deep learning (ours) |
| 2 | How damaged is each building? | Siamese crop classifier with a pretrained EfficientNetB0 encoder | Deep learning (ours) |
| 3 | What is each building made of? | The region's construction mix from the GEM Global Exposure Model ("Layer 1") | Lookup |
| 4 | How much material is in it? | RASMI material intensities (kg per m² of floor) | Lookup |
| 5 | How much embodied carbon to rebuild? | Two routes: RASMI × ICE carbon factors, and GEM's own embodied-carbon figures | Lookup |
| 6 | How much would rebuilding cost? | GEM building replacement cost per m² | Lookup |

**Design principle.** Only steps 1 and 2 are learned. Everything after the damage class is a transparent calculation where every assumption is an explicit table in the notebook, so a non-expert can follow one building from image to tonnes of CO2e. Every notebook section ends with a visible output (a chart, table, image or metric) so nothing happens silently.

**What the numbers mean (always stated with them).** "Carbon" means *embodied material carbon*: the emissions from producing the materials needed to rebuild. It excludes transport, construction-site energy, demolition and waste. "Cost" means *building replacement cost* from GEM: the cost to construct the building as built, structural and non-structural parts, **including labour**, **excluding contents**. Both are regional averages per m², applied to each building.

---

## 2. Data and the rules for using it

### 2.1 Sources

| Source | What we use from it | Licence |
|---|---|---|
| **xBD** (Gupta et al., 2019) | 1024 × 1024 pre/post satellite image pairs, building outlines (polygons), 4-level damage labels | CC BY-NC-SA 4.0 |
| **xView2 baseline** (CMU, 2019) | Pretrained building-finding U-Net and damage classifier: our "before" | BSD-3 |
| **GEM Global Exposure Model** v2026.0.0 | Per state/province: building counts by occupancy and construction class, floor area, storeys, replacement cost, embodied carbon | CC BY-NC-SA 4.0 |
| **RASMI** (Fishman et al., 2024) | kg of 8 materials per m² of floor, by world region, structure type and function | CC BY 4.0 |
| **ICE Database Educational V5.0** (Circular Ecology, June 2026) | kgCO2e per kg of each material | Educational use only; may not be redistributed (so the repository contains no ICE values) |
| **EfficientNetB0 ImageNet weights** (Tan & Le, 2019) | Starting weights for our pretrained encoders | Apache 2.0 |

USGS commodity prices were in the original plan for cost but were dropped (8 October) in favour of GEM, because GEM gives a full replacement cost per m² for every region and construction class, not just raw material prices.

### 2.2 The xBD splits

| Split | Tiles | Labelled buildings | Used for |
|---|---|---|---|
| `train` | 2,799 | 159,794 | Training; **224 tiles (13,399 buildings) held back as validation** |
| `tier3` | 6,369 | 144,576 | Extra training data only (nine further disasters) |
| `test` | 933 | 53,850 | Comparing models and methods |
| `hold` | 933 | 53,137 | **Used once, at the very end**, after everything was frozen |

Buildings labelled `un-classified` (no damage label) are excluded from training and scoring. They count as buildings in the building map.

### 2.3 The rules that keep the results honest

1. **Validation chooses, test reports.** Every choice (which model, which training setting, the watershed setting) was made using the validation tiles. The test split was only used to report how well the chosen thing works. Choosing by test score would make the test score optimistic.
2. **Validation is split by tile, not by building.** Buildings in the same image share lighting, season and neighbourhood. If some went to training and some to validation, the validation score would be inflated. 10% of training *tiles* (seed 0) form the validation set.
3. **The hold-out set is used exactly once.** It stayed in its archive until every choice was frozen (Section 8 lists them), then the final pipeline ran on it once. Whatever it showed is reported as it is; nothing was changed in response.

### 2.4 Two things found in the data (Section 1)

- **The `gsd` field in the labels is not the image's pixel size.** Each building is stored twice, once in pixel coordinates and once in longitude/latitude, so dividing real area by pixel area gives the true pixel size: about **0.47 m per pixel**, while `gsd` says about 2.5. Using `gsd` would have overstated every footprint by roughly 25–30 times. Footprints are therefore computed from the longitude/latitude polygons (`footprint_m2`).
- **`nepal-flooding` is in India.** Its building coordinates (about 26.6°N, 83.4°E, near Gorakhpur) lie in Uttar Pradesh, India. All 19 disaster locations were checked against building coordinates, and this one was corrected to use India's GEM and RASMI figures.

---

## 3. The baseline we compare against (Sections 2–3)

The xView2 baseline is the official starting model of the xView2 challenge. It has two separate parts, and both were reproduced exactly so that every improvement is measured against the real baseline, not a guess at it.

### 3.1 Building finder: a U-Net (Section 2)

A U-Net shrinks the image step by step to find what is in it (the *encoder*), then grows it back to full size to mark where (the *decoder*), with *skip connections* that hand fine detail from each encoder level to the decoder. It outputs a building / background score for every pixel.

**What we changed from the GitHub code, and why:** the baseline U-Net is written in Chainer, a framework that is no longer maintained. We rebuilt the identical architecture in Keras and copied the original trained weights into it (re-ordering each weight array, because Chainer stores convolution kernels as out × in × height × width and Keras as height × width × in × out). Nothing was retrained.

**Baseline scores on the test split:** precision 0.946, recall 0.699, **IoU 0.673, F1 0.804**. It rarely marks something that is not a building, but misses about 30% of building pixels.

### 3.2 Damage classifier (Section 3)

For each building it cuts a crop around the building (the bounding box grown by 80% on each side for context), resizes it to 128 × 128, and passes it through two branches (a small CNN and a ResNet50) whose outputs are joined and classified into the 4 damage classes.

**Important detail:** the baseline only ever looks at the **post**-disaster crop. It cannot compare before and after.

**What we changed, and why:**

| Change | Why |
|---|---|
| Weights loaded layer by layer with a name translation | The file was saved with 2019 Keras, which named ResNet50 layers differently |
| Crops cut in memory, not written to disk | Same crops, far less disk space |
| Blue and red channels swapped | *Reproduces a baseline quirk:* it saved its crops with OpenCV (blue-green-red order), so it was trained on swapped colours |
| Pixels multiplied by 1.4 | *Reproduces a baseline quirk:* its published inference script does this. With the "correct" 1/255 scaling it predicts no-damage for every building |

**Baseline scores on the test split:** macro F1 **0.311**; per class no-damage 0.787, minor 0.081, major 0.030, destroyed 0.344; 22% of predictions off by two or more damage levels. It almost never predicts major damage and calls 18% of undamaged buildings destroyed.

### 3.3 How models are scored (used throughout)

| Metric | Meaning |
|---|---|
| **F1 per class** | Combines precision and recall for one class; 1 = perfect |
| **Macro F1** | Plain average of the four class F1s, so rare classes count as much as common ones. **Our main measure.** Accuracy alone would mislead: answering "no-damage" for everything scores 77% |
| **xView2 F1** | Harmonic mean of the four class F1s: the official challenge score; punishes the weakest class |
| **Quadratic weighted kappa (QWK)** | Agreement that penalises errors by the square of how many levels they are off; suits ordered classes |
| **Far errors** | Share of predictions two or more damage levels away from the truth: the errors that hurt carbon estimates most |
| **IoU / F1 (pixels)** | For building maps: overlap of predicted and true building pixels |

---

## 4. Building our damage model (Sections 3b–3d)

This is the project's main deep-learning contribution. All experiments use the same data, the same validation tiles and the same 53,850 test buildings, so their scores can be compared directly. They are listed in one table in the notebook (`P3_EXPERIMENTS`).

### 4.1 What the model sees (Section 3b)

For every labelled building, a **pair** of crops: the same 64 × 64 window cut from the before image and from the after image, stacked into 6 channels. Damage is a *change*, so the model should see both.

| Choice | Value | Why |
|---|---|---|
| Crop size | 64 × 64 | A typical building is 20–40 pixels wide; 64 holds it with its surroundings, and is 4 times cheaper than the baseline's 128 |
| Margin | 80% of the building's size per side | Same as the baseline, so the comparison is about the model, not the crop |
| Colours | True colours, bilinear resizing | The baseline's swapped colours and ×1.4 were quirks, not design choices |

### 4.2 The model: a Siamese network (Section 3c)

One encoder (a stack of convolution layers turning a crop into a vector of numbers) is applied **twice with the same weights**: once to the before crop and once to the after crop. The head then receives three things: the before features, the after features, and **their difference**, so "what changed" is given to it directly. A dense layer with dropout then outputs 4 class scores.

Two encoders were used:
- **From scratch:** 4 blocks of two 3×3 convolutions (batch normalisation, ReLU), channels 16 → 128; about 0.34 million parameters, against 124 million in the baseline.
- **Pretrained:** EfficientNetB0 trained on ImageNet (about 4.5 million parameters for the whole pair model), which already recognises edges, textures and shapes. It starts at a lower learning rate (0.0003) so the pretrained weights are adjusted gently.

**Training setup:** Adam optimiser; batches of 128; random flips and 90° rotations applied identically to before and after; after every epoch the model is scored on the validation tiles and **the weights from the best validation epoch are kept**. Each epoch draws a fresh random sample of training buildings.

### 4.3 The experiments

Each experiment changes one thing so its effect can be read off.

**Losses compared** (all treat the same four classes):

| Loss | What it does | Why try it |
|---|---|---|
| Cross-entropy | Standard classification loss | Reference |
| Class-weighted cross-entropy | Each building weighted inversely to how common its class is (no-damage about 0.33, damaged classes about 2.7–3.5) | 73% of buildings are undamaged; stops them dominating |
| Focal loss (γ = 2) | Shrinks the loss on examples the model already gets right | Another standard fix for imbalance |
| Ordinal (CORAL) | One "damage score" plus three thresholds (≥ minor? ≥ major? destroyed?) | Damage levels are ordered; an error of three levels should cost more than one |

**Set 1: train split only, small budget** (10 epochs × 40,000 buildings, on a laptop CPU)

| Experiment | Test macro F1 | Carbon predicted ÷ true |
|---|---|---|
| Cross-entropy | 0.628 | 0.56 |
| Class-weighted | 0.624 | 1.06 |
| Focal | 0.595 | 1.51 |
| Ordinal | 0.474 | 1.58 |

**Set 2: train + tier3, larger budget** (20 epochs × 150,000 buildings, Colab GPU)

| Experiment | Validation macro F1 | Test macro F1 | Carbon ratio |
|---|---|---|---|
| Cross-entropy | 0.682 | 0.705 | 0.95 |
| Class-weighted | 0.674 | 0.686 | 1.40 |
| Focal | 0.660 | 0.678 | 1.07 |
| Ordinal | 0.574 | 0.573 | 1.18 |

**Finding:** more data and longer training improved every loss by 0.06–0.10. Validation scores were noisy (swinging by up to 0.14 between epochs), and each recipe had been run once.

**Set 3: answering what Set 2 left open** (train + tier3, 20 × 150,000)

| Experiment | Change | Validation | Test macro F1 | Carbon ratio |
|---|---|---|---|---|
| CE, cosine | learning rate falls smoothly to zero | 0.701 | 0.743 | 0.86 |
| CE, cosine, seed 1 | same, different random start | 0.693 | 0.735 | 0.85 |
| Focal, cosine | | 0.668 | 0.691 | 1.18 |
| Focal, cosine, seed 1 | | 0.662 | 0.674 | 1.25 |
| **Pretrained, CE, cosine** | EfficientNetB0 encoder | **0.712** | **0.758** | **1.01** |
| Pretrained, focal, cosine | | 0.666 | 0.700 | 1.37 |

**What Set 3 showed:**
1. **The cosine schedule fixed the noisy training** (validation curves now settle) and added about 0.04.
2. **The gap between losses is real.** Two seeds of the same recipe differ by about 0.01–0.02; plain cross-entropy beats focal by about 0.05. Focal loss consistently over-predicts damage.
3. **Pretraining helps a little:** about +0.015 over the scratch encoder, roughly one to two times the seed-to-seed spread, from one seed.

### 4.4 The selected model

**Selected by validation score:** the pretrained encoder with plain cross-entropy and the cosine schedule (`effnet_ce+tier3_cosine`). It was best on validation (0.712) and, separately, best on test: macro F1 **0.758**, per class 0.95 / 0.59 / 0.69 / 0.80, QWK 0.827, far errors 3.8%, carbon within 1% of the true-label figure.

**Why class weighting and focal loss were not used** even though imbalance is real: with enough data, plain cross-entropy scored higher on validation, and both imbalance methods pushed the model to over-predict damage, inflating the carbon estimate (focal +18% to +37%).

### 4.5 Why the ordinal model was weak (Section 3d)

Expected to help, ordinal loss scored worst. We tested whether its *cut-offs* were the problem (class weights could push all three thresholds towards "damaged") by moving only the three cut-offs to their best positions on validation, with no retraining. Test macro F1 rose only from 0.573 to 0.593. So the cut-offs were not the main problem; the model's underlying ranking of damage (especially minor vs major) is weaker. Fair wording: "ordinal loss did not help at this budget", not "ordinal loss cannot help".

### 4.6 The multi-task model (Section 3c-ii)

The design used by the leading xView2 entries: **one network that reads the whole 1024 × 1024 before/after tile at full resolution** and draws a building map and a damage map at once. It is a Siamese U-Net: the pretrained EfficientNetB0 encoder reads both images; at five scales the before features, after features and their difference are joined; the decoder rebuilds full resolution and outputs 1 building score and 4 damage scores per pixel.

- **Loss:** building map = binary cross-entropy + Dice (Dice scores the overlap of the whole predicted and true building areas, so the huge background does not dominate); damage map = cross-entropy on labelled building pixels only.
- **Training:** random 512 × 512 windows, batch 8, 30 epochs × 500 steps, cosine from 0.0003, mixed precision on an A100 (2 h 54 min).
- **Fair comparison:** each surveyed building gets the damage class predicted for the majority of its pixels, so it is scored on exactly the same buildings as the crop models.

**Result:** damage macro F1 0.658 validation, 0.667 test, so **it did not beat the crop model** (0.758). Its weak class was minor damage (F1 0.33), which covers few pixels and looks subtle. It was still improving at epoch 30. **But its building map is better than the baseline's:** IoU 0.686 vs 0.673, recall 0.834 vs 0.699 (it finds far more buildings, with more false alarms). So the final pipeline uses **its building map with the crop model's damage judgements**.

A technical note: the first Colab run failed because TensorFlow's XLA compiler rejected the mix of 16-bit and 32-bit numbers that mixed precision creates in the bilinear up-sampling step. Fixed by turning the compiler off for this model (`jit_compile=False`).

---

## 5. Construction type (Section 4)

Materials depend on what a building is made of, which a satellite image does not directly show. The original plan used a hand-written rule (building density → type). It was replaced (8 October) because it left 27% of tiles unclassified and had no source.

### 5.1 Layer 1: the regional mix from GEM (used for all headline results)

GEM lists, for each state or province, how many buildings of each occupancy (residential, commercial, industrial) and construction class exist, with their storeys, floor area, cost and embodied carbon. Each building is given **the mix of its region**: we do not claim to know that one Texas house is timber; we say it is, for example, 64% likely timber-frame residential, 18% masonry residential, and so on, and compute with that weighted average. Over thousands of buildings, the regional total comes out right.

Three explicit assumption tables (Section 4):
1. **Disaster → GEM region → RASMI region** (`DISASTER_LOCATION`): each disaster is matched to the state/province holding most of its buildings, checked against coordinates.
2. **GEM construction class → RASMI structure type** (`GEM_TO_RASMI_STRUCTURE`): wood → timber; unreinforced, reinforced/confined masonry, adobe/stone/earth and India's hybrid class → masonry; reinforced concrete → concrete; steel → steel; GEM's "other" (mobile homes, informal housing) → timber (an assumption; tested in the sensitivity analysis).
3. **GEM occupancy → RASMI function:** non-residential → NR; residential with an average of 3 or more storeys → multi-family (RM), otherwise single-family (RS).

**Check:** GEM's implied average footprint per building was compared with the footprints measured in xBD. Medians match for about half the disasters; xBD buildings run larger in US suburbs (1.3–1.5×). This led to reporting two floor-area methods (Section 6.3).

### 5.2 Layer 2: a CNN trained from regional proportions (Section 4b): a negative result

**Idea:** a small CNN looks at each building's before crop and footprint and outputs its own probabilities of timber / masonry / concrete / steel. There are no per-building labels, so it was trained by **learning from label proportions**: for a batch of 256 buildings from one region, the *average* of its predictions must match GEM's mix for that region (loss = KL divergence).

**Test: leave one region out.** For each of the 16 regions, train on the other 15 and predict the mix of the unseen one. Compare its error (total variation distance) with a **no-image guess** (the average mix of the 15 training regions). A model that cannot beat that guess has learned nothing transferable.

**The flaw found and fixed.** The first version used *batch normalisation*, which rescales each layer using statistics of the whole batch. Because every batch came from a single region, the batch statistics themselves revealed the region, so the network could match the regional mix during training without learning anything about buildings, and behaved differently when used normally. This was caught because the all-regions model was far from GEM's mix on its *own training regions* (0.54 total variation). It was fixed by switching to *group normalisation* (each image normalised by its own values only), and the earlier results were discarded.

**Valid result:** the CNN beat the no-image guess in **0 of 16** regions (mean error 0.343 vs 0.311). It predicted almost the same mix for every building everywhere: it had learned roughly the global average. With only 16 regions of supervision, the average is the easiest solution. Fair wording: "as built and at this budget, learning from 16 regional proportions gave no usable per-building signal". Layer 1 stays the method.

### 5.3 Within-region allocation (Section 4c, 9c)

Two alternatives to giving every building the same mix were also compared (both rescaled so each region's average still equals GEM's mix exactly):
- **Size-adjusted Layer 1 (no training):** shifts each building's mix by its footprint, using GEM's average building size for each construction type in that region (larger-than-average buildings are more likely to be the larger types). One judgement setting: sizes within a type are assumed to scatter by about a factor of 2.
- **Layer 2 CNN trained on all regions.**

**Result:** the size-adjusted version raises carbon by about 22% (cost +6%), because heavier construction moves onto larger buildings. That allocation matters about as much as the RASMI range. Layer 2 changed totals by less than 1% (its predictions were nearly constant). Layer 1 remains the headline; size-adjusted is reported as an alternative.

---

## 6. Materials, carbon and cost (Sections 5–7)

### 6.1 The calculation for one building

```
rebuilt footprint  = footprint (m²) × rebuild % for its damage class
rebuilt floor area = rebuilt footprint × average storeys of the region's mix
material mass      = rebuilt footprint × Σ over the region's building classes of
                     (share of buildings × storeys × RASMI kg per m² of floor)
carbon, route A    = Σ over materials of mass × ICE carbon factor        (RASMI × ICE)
carbon, route B    = rebuilt floor area × GEM embodied carbon per m²     (GEM)
replacement cost   = rebuilt floor area × GEM replacement cost per m²    (GEM)
```

Everything after "rebuilt footprint" depends only on the region, so it is computed once per region as a table of "what 1 m² of footprint means here" (`regional_factors()`), and each building multiplies its own rebuilt footprint by that row.

### 6.2 The assumptions

| Assumption | Value | Why / status |
|---|---|---|
| Rebuild % by damage class | none 0%, minor 15%, major 60%, destroyed 100% | Stated assumption (no cited source); varied ±20% in the sensitivity analysis |
| RASMI central value | Median (50th percentile) | RASMI publishes a skewed range; median is robust; 25th/75th used for sensitivity |
| ICE factor per material | One general entry per material (e.g. "Concrete, average UK mix", "Steel, Rebar" world average, "Timber, average of all data") | Specific product unknown, so general/average entries. Exact entries and sheet rows are in `tools/build_ice_factors.py` |
| Timber carbon | ICE timber factor **excluding** carbon stored in the wood | Storage credits are valid only for sustainably sourced timber, which cannot be assumed after a disaster; the including-storage factor is a sensitivity case |
| Copper | Excluded | ICE V5.0 withdrew its copper factors; copper is about 0.2 kg per m² of floor, negligible |
| `un-classified` buildings | Excluded | No damage class to apply a rebuild % to |

### 6.3 Two carbon routes and two floor-area methods

**Why two carbon routes:** route A shows *which materials* the carbon comes from and is auditable line by line; route B is an independent figure from a different source and method. Agreement means the estimate is in the right range; disagreement shows the uncertainty. On the training split with true labels, route A is **0.73 of route B** overall (0.50 Palu to 1.20 Haiti). Route A by material: brick 29%, concrete 25%, steel 18%, timber 18%.

**Why two floor-area methods:** floor area is the assumption the totals are most sensitive to. *Measured footprint* (each building's own outline × regional storeys) uses what is seen but traces roofs, which overhang walls. *GEM average size* gives every building its region's average, consistent with the cost/carbon source but ignoring real size. They differ by about 40% (1.13 vs 0.65 Mt CO2e on the training split), so **both are reported side by side** (Section 7c). The measured-footprint method is the one used in the pipeline tables.

---

## 7. How errors in the damage model flow into carbon and cost (Sections 7b, 9a)

The same calculation was run on the test buildings with the true damage labels and with each model's predicted labels. The ratio (predicted ÷ true) shows how much a model's errors distort the totals; 1.00 is ideal.

| Model | Test macro F1 | Carbon ratio | Cost ratio |
|---|---|---|---|
| Baseline | 0.311 | 2.31 | 1.7 |
| Our CE, train only | 0.628 | 0.56 | 0.62 |
| **Selected model** | **0.758** | **1.01** | **1.03** |

Two lessons for the report:
- **Two models with similar F1 can give very different carbon totals** (CE vs class-weighted on Set 1: same F1, 0.56 vs 1.06), because what matters downstream is the *direction* of the errors.
- **An accurate total can hide large per-disaster errors.** For the selected model on test, per-disaster carbon ratios range from about 0.9 to 1.2 for most disasters, but a disaster with almost no damage (Mexico City earthquake, 1% of buildings damaged) is far off in relative terms. Misses and false alarms partly cancel in the sum.

---

## 8. Sensitivity analysis (Section 9b)

Each assumption changed alone, on the training split with true labels; % change in route A carbon:

| Change | Route A carbon | Route B carbon | Cost |
|---|---|---|---|
| Rebuild % −20% | −20% | −20% | −20% |
| Rebuild % +20% (capped at 100%) | +12.5% | +12% | +12% |
| RASMI 25th percentile | −36% | — | — |
| RASMI 75th percentile | +49% | — | — |
| GEM "other" as masonry | +4.5% | — | — |
| Floor area from GEM averages | −42% | −44% | −41% |
| Timber including carbon storage | −54% | — | — |

**Reading it:** the floor-area method, the timber storage choice and the RASMI range move the totals most; the uncertain "other" mapping barely matters. The headline figures should be read as a central estimate within roughly a factor of 1.5–2.

---

## 9. The prototype: running with no human help (Section 9d)

Everything above that used a model was given the **surveyed building outlines**. A real user has none. The prototype (`prototype.ipynb`, using `xbd_pipeline.py`) runs the whole chain from just two images.

**How it is checked:**
1. **The module reproduces the notebook:** identical carbon and cost totals (0.0% difference) and the same damage answer on 1,500 of 1,500 test buildings. (A bug was found here: EfficientNet takes its input-normalisation constants from the ImageNet file and they are not saved with our weights, so the model must be built from ImageNet weights *before* our weights are loaded. Without that, only ~75–80% of answers matched.)
2. **The whole chain on the test split,** compared with true labels.

**The merging problem and the watershed fix.** The first automatic run over-estimated carbon **1.77 times**, although the damage model alone was within 1%. The cause: the building map marks *pixels*, so buildings that touch merge into one large patch (up to 100,000 m² in Mexico City), and if any part looks damaged, the whole patch is "rebuilt". It found only 26,563 buildings against 53,850 surveyed.

A **watershed** splits patches without training: each building pixel's distance to the nearest non-building pixel peaks in the middle of a building and dips at the neck where two touch; peaks at least *N* pixels apart become seeds, and the patch is shared out between them along the dips. *N* was chosen on the **validation tiles** from {none, 3, 5, 8, 12, 16} by the typical per-disaster carbon error. All settings scored almost the same on validation (1.20–1.21), so the choice (**12**) is weakly supported; it gave the building count closest to the survey.

**Test result:** automatic carbon ratio 1.77 → **1.53**; buildings found 26,563 → 47,776. Merging was only part of the over-estimate; the rest comes from the building finder's false alarms (about one marked pixel in five is not a building) and test being harder than validation. No further tuning was done on test.

---

## 10. Final hold-out evaluation (Sections 8 and 10)

**Frozen before the run (Section 8):** damage model `effnet_ce+tier3_cosine`; building finder = multi-task U-Net; watershed separation 12; construction type = Layer 1; all assumption tables as stated. The hold-out archive was unpacked for the first time by Section 10 and run once.

**Damage (surveyed outlines, 53,137 buildings):**

| | Baseline | Final model (hold-out) | Final model (test) |
|---|---|---|---|
| Macro F1 | 0.317 | **0.743** | 0.758 |
| F1 no / minor / major / destroyed | 0.77 / 0.10 / 0.03 / 0.37 | 0.93 / 0.60 / 0.66 / 0.78 | 0.95 / 0.59 / 0.69 / 0.80 |
| QWK | 0.347 | 0.802 | 0.827 |
| Off by 2+ levels | 21.8% | 4.5% | 3.8% |

The final model scores almost as well on completely unseen data as on test, so the test result was not a product of model selection.

**Finding buildings:**

| | Precision | Recall | IoU | F1 |
|---|---|---|---|---|
| Baseline U-Net | 0.950 | 0.707 | 0.681 | 0.810 |
| Multi-task U-Net | 0.798 | 0.841 | 0.693 | 0.819 |

**Carbon (route A) as a multiple of the true-label figure, all 10 disasters:**

| | Overall | Range across disasters |
|---|---|---|
| Assisted (surveyed outlines + model damage) | **1.03** | 0.64 – 3.90 |
| Automatic, no splitting | 1.51 | 0.87 – 8.94 |
| Automatic, split (the prototype) | **1.41** | 0.88 – 6.15 (cost 1.29) |

**How to read these:** with surveyed outlines, the carbon estimate is accurate in aggregate (3% high). Fully automatic, it gives the right order of magnitude but over-estimates by about 40%, and badly in dense urban areas (Palu 2.07×, Mexico City 6.15×) where touching buildings are hard to separate. Suburban disasters are reasonable (SoCal 0.88, Michael 1.18, Harvey 1.29). Large *relative* errors for Mexico City partly reflect its tiny true total (2.3 kt).

---

## 11. Limitations

1. **Construction type is a regional mix, not a per-building fact.** There is no per-building construction ground truth in xBD; Layer 1 can only be validated at the regional level.
2. **Each disaster is matched to one state/province,** and GEM figures describe the whole state, not the neighbourhoods photographed.
3. **The GEM → RASMI mapping is an assumption,** especially for GEM's "other" and India's "hybrid" classes.
4. **RASMI values are medians of ranges;** some regions have few source buildings and are filled from wider pools.
5. **Rebuild percentages are assumed.**
6. **Floor area is uncertain** (two methods, ~40% apart); storeys are regional averages, not observed.
7. **Carbon is embodied material carbon only;** copper excluded.
8. **Cost is a regional average replacement cost,** including labour, excluding contents; not a quote or an insured/economic loss.
9. **Damage is judged from overhead images only;** minor damage is the hardest class (F1 ~0.6).
10. **The automatic pipeline over-estimates in dense areas** (merged buildings, false alarms).
11. **Most experiments were run once with one seed;** the seed repeats in Set 3 show a spread of about 0.01–0.02 macro F1.
12. **Aggregate totals are more reliable than any single disaster or building,** because errors partly cancel.

---

## 12. Decisions log

| Date (2026) | Decision | Why |
|---|---|---|
| 8 Oct | Compute footprints from longitude/latitude polygons, not `gsd` | `gsd` is not the image pixel size (0.47 m measured vs ~2.5) |
| 8 Oct | Reproduce the baseline exactly (Chainer → Keras port; keep its colour/scale quirks) | Improvements must be measured against the real baseline |
| 8 Oct | Use ICE Educational V5.0 (not v4.1) | The version available |
| 8 Oct | Drop USGS; cost from GEM replacement cost | GEM gives full replacement cost per m² for every region and class |
| 8 Oct | Two-layer construction-type design; drop the density rule table | Rule table had no source and left 27% of tiles unclassified |
| 8 Oct | Compute carbon by both routes | Independent cross-check |
| 8 Oct | RASMI median as central value | Robust to skewed ranges |
| 8 Oct | Exclude copper | No ICE V5.0 factor |
| 8 Oct | `nepal-flooding` uses India's figures | Building coordinates are in Uttar Pradesh |
| 8 Oct | Timber excludes stored carbon | Sustainable sourcing cannot be assumed |
| 8 Oct | Report both floor-area methods | ~40% apart; no basis to pick one |
| 8 Oct | Add tier3 as extra training data | Nearly doubles the training buildings and adds nine disasters |
| 10 Oct | Select models by validation score only | Keeps test scores honest |
| 10–15 Oct | Run Set 3 (cosine, seeds, pretrained) | Settle noisy training and test whether differences are real |
| 11 Oct | Layer 2 reported as a negative result; Layer 1 stays | 0 of 16 regions beaten after fixing the batch-norm flaw |
| 15 Oct | Final damage model: pretrained crop model | Best validation score; multi-task model did not beat it |
| 15 Oct | Prototype: multi-task building map + crop-model damage | Each part best at its own job |
| 18 Oct | Rebuild percentages stay as stated assumptions | No source; sensitivity analysis shows their effect |
| 18 Oct | Watershed splitting, separation 12, chosen on validation | Reduces merging; test 1.77 → 1.53 |
| 18 Oct | Freeze everything; run the hold-out once | Final honest estimate |

---

## 13. Likely questions and answers

**What did you build yourselves, and what did you reuse?**
We built and trained the damage classifiers (Siamese crop models, from scratch and with a pretrained encoder), the multi-task Siamese U-Net, the Layer 2 CNN, the watershed splitting, the whole carbon/cost pipeline and all evaluation. We reused the xView2 baseline only as the comparison point (re-implemented in Keras with its released weights) and ImageNet weights as a starting point for EfficientNet.

**Why is macro F1 your main metric and not accuracy?**
73% of buildings are undamaged, so a model that always says "no damage" reaches high accuracy while being useless. Macro F1 weights all four classes equally.

**How do you know the test score is not inflated by trying many models?**
Models were chosen by validation score only, and the final model was then evaluated once on the untouched hold-out set: 0.743 there vs 0.758 on test.

**Why did plain cross-entropy beat the imbalance methods?**
With 291,000 training buildings, the minority classes had enough examples. Class weighting and focal loss shifted predictions towards damage, which raised minor/major recall a little but cost more in precision; validation scores were lower and carbon was over-estimated. The second seed confirmed the gap is larger than run-to-run noise.

**Why Siamese, and why the difference of features?**
Damage is a change between before and after. Sharing the encoder means both images are described in the same terms, and handing the head the difference makes "what changed" explicit.

**Why 64-pixel crops?**
Most buildings are 20–40 pixels wide at 0.47 m/pixel; 64 holds the building with context and is four times cheaper than 128, which made laptop and free-Colab training feasible.

**The multi-task model is what the winners used; why didn't it win here?**
Single model, 30 epochs (still improving), no ensembling. It is weak on minor damage, which occupies few pixels. Its building map, however, is better than the baseline's, so it is used for finding buildings.

**What is Layer 2 and why does it not work?**
A CNN trained only from regional construction mixes (no per-building labels). With 16 regions, the easiest solution is to predict the average mix everywhere, which is what it learned; it beat a no-image guess in 0 of 16 held-out regions. We also found and fixed a batch-normalisation flaw that would have made an earlier version look better than it was.

**Why two carbon routes, and why do they disagree?**
They are independent sources and methods. Route A (RASMI × ICE) is about 0.73 of route B (GEM) overall. Differences come from material intensities, which materials are counted, and the carbon factors; the spread is a measure of uncertainty.

**Why not use the timber carbon-storage credit?**
It is only valid if the timber is from sustainably managed forests. Post-disaster reconstruction timber cannot be assumed to be; including it would cut route A carbon by 54%. It is shown as a sensitivity case.

**Your carbon estimate is within 3% on the hold-out. Is it that accurate?**
In aggregate, with surveyed outlines, yes, but partly because errors cancel. Single disasters range from 0.64 to 3.90 times the true figure. Fully automatic it is 1.41 times. And the "true" figure itself depends on the stated assumptions (rebuild %, RASMI, ICE, GEM); the sensitivity analysis shows those move it by tens of per cent.

**Why does the automatic pipeline over-estimate?**
Touching buildings merge into large patches, and the building finder marks some non-building pixels. Watershed splitting reduced the error from 1.77× to 1.53× on test (1.41× on hold-out).

**Why not tune the watershed on test to get closer to 1.0?**
Then the test result would no longer be an honest measure. It was tuned on validation only.

**Where do the rebuild percentages come from?**
They are stated assumptions (0/15/60/100%). The model decides the damage class; the rebuild % turns that class into how much of the building is rebuilt. The sensitivity analysis shows ±20% moves totals by −20%/+12.5%.

**Can someone reproduce this?**
Yes: the repository has all code, the reference tables and setup instructions; trained models and cached results are on the Releases page; seeds are fixed; ICE values are rebuilt from the user's own download.

---

## 14. Where everything is in the code

### Main notebook (`xBD_damage_to_carbon_cost.ipynb`)

| Section | What | Key names |
|---|---|---|
| 0 | Config: all paths and switches in one cell | `XBD_ROOT`, `OUTPUT_DIR`, `TEST_TILE_LIMIT` |
| 1 | Load labels, class balance, per-disaster counts, pixel-size check | `build_metadata_table`, `buildings_df` |
| 2 | Baseline U-Net (Keras port), test scores | `build_localization_unet`, `load_chainer_weights`, `localization_scores` |
| 3 | Baseline classifier, confusion matrix, metrics | `build_baseline_classifier`, `crop_building`, `damage_scores`, `results_table` |
| 3b | Crop-pair caches, validation split by tile | `build_crop_cache`, `val_tiles` |
| 3c | Siamese models, losses, data batches, experiment table, training loop | `build_pair_classifier`, `focal_loss`, `coral_loss`, `CropBatches`, `CropStore`, `P3_EXPERIMENTS`, `run_experiment` |
| 3c-ii | Multi-task U-Net: tile export, model, loss, training | `export_tiles_for_colab`, `build_multitask_unet`, `multitask_loss`, `run_multitask` |
| 3c (results) | Model selection by validation, per-disaster scores | `selection`, `selected_model_name` |
| 3d | Ordinal cut-off calibration | `calibrate_cutoffs` |
| 4 | Layer 1 tables and regional mix | `DISASTER_LOCATION`, `GEM_TO_RASMI_STRUCTURE`, `rasmi_function`, `region_classes` |
| 4b | Layer 2 (label proportions, leave-one-region-out) | `build_structure_classifier`, `proportion_loss`, `RegionBags` |
| 4c | Within-region allocation, size check | `fit_to_region_mix`, `size_probs`, `implied_rel_size` |
| 5 | Rebuild %, RASMI, per-footprint factors, material mass | `REBUILD_PCT`, `regional_factors`, `add_material_mass` |
| 6 | Carbon, both routes | `add_carbon`, `ICE_CARBON_FACTORS` |
| 7 | Cost; error propagation (7b); two floor-area methods (7c) | `add_cost`, `reconstruction_totals`, `FLOOR_AREA_METHODS` |
| 8 | Hold-out protocol (frozen choices) | — |
| 9a–9c | Dashboard, sensitivity, allocation effect | `scenario_totals`, `totals_with_allocation` |
| 9d | Prototype checks, watershed tuning, automatic runs | `xp` (the module), `SPLIT_CANDIDATES`, `CHOSEN_SEPARATION` |
| 10 | The one-off hold-out run | `hold_final`, `hold_four` |
| End | Limitations and decisions log | — |

Slow steps are cached in `outputs/` (predictions, histories, weights). Delete a cache file to recompute that step. `P3_TRAIN`, `MT_TRAIN` and `L2_TRAIN` control whether missing experiments are trained.

### Other files

| File | What |
|---|---|
| `xbd_pipeline.py` | The finished pipeline: `load_models`, `find_buildings` (with `split_touching`), `judge_damage`, `region_factors`, `estimate`, `run` |
| `prototype.ipynb` | Runs the pipeline on one image pair, one output per step |
| `colab_train.ipynb`, `colab_train_multitask.ipynb` | GPU training on Colab; generated from the notebook's cells by `tools/make_colab_notebook.py`, so the code is identical |
| `tools/build_ice_factors.py` | Builds the ICE table from your own ICE download, checking each row's name |
| `tools/export_public_repo.py` | Builds the public repository from an allow-list of files, removes ICE outputs and local paths, and refuses to finish if any remain |

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Siamese network** | One network applied to two inputs with shared weights, so both are described the same way |
| **Encoder / decoder** | The part that turns an image into compact features / the part that turns features back into a full-size map |
| **U-Net** | Encoder–decoder with skip connections that pass fine detail across, used for pixel maps |
| **Pretrained (ImageNet)** | Starting from weights learned on 1.2 million everyday photos, then fine-tuning |
| **Epoch** | One pass of training (here: a fixed number of randomly drawn examples) |
| **Cosine schedule** | Learning rate that decreases smoothly to zero over training |
| **Batch / group normalisation** | Rescaling layer outputs using statistics of the whole batch / of each image alone |
| **Focal loss** | Loss that down-weights examples already classified correctly |
| **CORAL (ordinal loss)** | One score plus ordered thresholds, for ordered classes |
| **Learning from label proportions** | Training with only group-level class proportions, not individual labels |
| **IoU** | Intersection over union: overlap ÷ combined area |
| **Total variation distance** | Half the sum of absolute differences between two mixes; 0 = identical, 1 = completely different |
| **Watershed** | Splitting a shape by flooding from seed points along its low ridges |
| **Embodied carbon** | Emissions from producing building materials |
| **Replacement cost** | Cost to rebuild a building as it was (structure + non-structural, incl. labour, excl. contents) |
