"""The finished damage -> carbon -> cost pipeline, for use on new image pairs.

This module holds ONLY what is needed to use the trained models (see prototype.ipynb).
How every piece was chosen, trained and tested is in xBD_damage_to_carbon_cost.ipynb,
whose Section 9d checks that this module reproduces that notebook's numbers.

Pipeline for one before/after image pair:
  1. find_buildings   - building map from the multi-task Siamese U-Net (Section 3c-ii),
                        split into separate buildings
  2. judge_damage     - damage class per building from the pretrained crop classifier
                        (Section 3c, experiment effnet_ce+tier3_cosine)
  3. estimate         - materials (RASMI), embodied material carbon (ICE and GEM) and
                        building replacement cost (GEM), Layer 1 regional mix (Sections 4-7)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import keras
from keras import layers

PROJECT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT / "outputs"
REFERENCE_DIR = PROJECT / "reference_data"

# ---------------------------------------------------------------------------- models
BUILDING_FINDER_WEIGHTS = OUTPUT_DIR / "mt_unet_effnet_30ep_500steps.weights.h5"
DAMAGE_MODEL_WEIGHTS = OUTPUT_DIR / "p3_effnet_ce+tier3_cosine_20ep_150000spe.weights.h5"
DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]   # ordinal order

CROP_SIZE = 64             # damage model input (pixels)
CROP_MARGIN = 0.8          # each building's box is grown by 80% per side for context
PIXEL_SIZE_M = 0.47        # metres per pixel of xBD imagery (measured in Section 1d); change for other imagery
MIN_BUILDING_PIXELS = 20   # blobs smaller than this (about 4.4 m2 at 0.47 m/px) are ignored as noise
SPLIT_MIN_SEPARATION_PX = 12     # watershed splitting of touching buildings (None = off). Chosen on the
                                 # 224 validation tiles in the main notebook, Section 9d (2026-10-18)

# ------------------------------------------------- assumptions (same as the main notebook)
REBUILD_PCT = {"no-damage": 0.0, "minor-damage": 0.15, "major-damage": 0.60, "destroyed": 1.0}
GEM_TO_RASMI_STRUCTURE = {"W": "T", "MUR": "M", "MR|MCF": "M", "ADO|ST|E": "M", "CR+": "C", "CR-": "C",
                          "S": "S", "OT": "T", "HYB": "M"}
MULTI_FAMILY_MIN_STOREYS = 3
EXCLUDED_MATERIALS = ["copper"]          # ICE V5.0 has no copper factor
COUNTRY_RASMI_REGION = {"USA": "OECD_USA", "HTI": "LAM_LAM-L", "GTM": "LAM_LAM-L", "MEX": "LAM_MEX",
                        "IDN": "ASIA_IDN", "IND": "ASIA_IND", "NPL": "ASIA_OAS-L", "PRT": "OECD_EU15",
                        "AUS": "OECD_AUNZ"}
DISASTER_REGION = {   # xBD disaster -> (country ISO3, GEM state/province), as in Section 4
    "guatemala-volcano": ("GTM", "Escuintla"), "hurricane-florence": ("USA", "North Carolina"),
    "hurricane-harvey": ("USA", "Texas"), "hurricane-matthew": ("HTI", "Grande’Anse"),
    "hurricane-michael": ("USA", "Florida"), "mexico-earthquake": ("MEX", "Ciudad de México"),
    "midwest-flooding": ("USA", "Oklahoma"), "palu-tsunami": ("IDN", "Sulawesi Tengah"),
    "santa-rosa-wildfire": ("USA", "California"), "socal-fire": ("USA", "California"),
    "joplin-tornado": ("USA", "Missouri"), "lower-puna-volcano": ("USA", "Hawaii"),
    "moore-tornado": ("USA", "Oklahoma"), "nepal-flooding": ("IND", "Uttar Pradesh"),
    "pinery-bushfire": ("AUS", "South Australia"), "portugal-wildfire": ("PRT", "Leiria"),
    "sunda-tsunami": ("IDN", "Banten"), "tuscaloosa-tornado": ("USA", "Alabama"),
    "woolsey-fire": ("USA", "California"),
}


# ------------------------------------------------------------------- model builders
# Architectures copied from the main notebook. The encoders are built from the ImageNet weights
# FIRST and our trained weights loaded on top: EfficientNet's built-in input normalisation takes its
# constants from the ImageNet file, and those constants are not stored in our saved weights. (Building
# the encoders without ImageNet weights reproduced only ~75-80% of the notebook's predictions.)
# Source of the ImageNet file: Keras Applications, efficientnetb0_notop.h5, downloaded once from
# https://storage.googleapis.com/keras-applications/ (Tan & Le 2019; Apache 2.0).
def _conv_block(x, channels):
    for _ in range(2):
        x = layers.Conv2D(channels, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return x


def build_building_finder():
    """Multi-task Siamese U-Net (main notebook cell mt03). Only its building map is used here."""
    before, after = keras.Input((None, None, 3), name="before"), keras.Input((None, None, 3), name="after")
    backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(None, None, 3))
    skips = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation",
             "block6a_expand_activation", "top_activation"]
    encoder = keras.Model(backbone.input, [backbone.get_layer(n).output for n in skips], name="shared_encoder")
    f_before, f_after = encoder(before), encoder(after)
    joined = [layers.Concatenate()([a, b, layers.Subtract()([b, a])]) for a, b in zip(f_before, f_after)]
    x = _conv_block(joined[-1], 256)
    for skip, channels in zip(reversed(joined[:-1]), [128, 64, 32, 16]):
        x = layers.UpSampling2D(interpolation="bilinear")(x)
        x = _conv_block(layers.Concatenate()([x, skip]), channels)
    x = layers.UpSampling2D(interpolation="bilinear")(x)
    maps = layers.Conv2D(5, 1, dtype="float32", name="maps")(x)
    return keras.Model([before, after], maps, name="multitask_unet")


def build_damage_model():
    """Crop-pair classifier with the pretrained EfficientNetB0 encoder (main notebook cell p3model02)."""
    pair = keras.Input((CROP_SIZE, CROP_SIZE, 6))
    encoder = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg",
                                                input_shape=(CROP_SIZE, CROP_SIZE, 3))
    before, after = encoder(pair[..., :3]), encoder(pair[..., 3:])
    h = layers.Concatenate()([before, after, layers.Subtract()([after, before])])
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.3)(h)
    return keras.Model(pair, layers.Dense(len(DAMAGE_CLASSES))(h))


def load_models():
    finder, damage = build_building_finder(), build_damage_model()
    finder.load_weights(BUILDING_FINDER_WEIGHTS)
    damage.load_weights(DAMAGE_MODEL_WEIGHTS)
    return finder, damage


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


# ------------------------------------------------------------- step 1: find buildings
def building_mask(finder, before, after):
    """True where the building finder marks a building pixel."""
    maps = finder.predict_on_batch((before[None].astype(np.float32), after[None].astype(np.float32)))[0]
    return maps[..., 0] > 0


def split_touching(mask, min_separation_px):
    """Number the buildings in a mask. Without splitting, every connected patch is one building.
    With splitting (a watershed): each pixel's distance to the nearest non-building pixel is
    largest in the middle of a building and dips at the narrow 'neck' where two buildings touch.
    Peaks of that distance at least min_separation_px apart become seeds, and each patch is
    divided between its seeds along the dips, like water filling neighbouring valleys."""
    if not min_separation_px:
        return ndimage.label(mask)[0]
    distance = ndimage.distance_transform_edt(mask)
    peaks = mask & (distance == ndimage.maximum_filter(distance, size=2 * min_separation_px + 1))
    seeds, n = ndimage.label(peaks)
    patches, _ = ndimage.label(mask)
    unseeded = mask & ~np.isin(patches, np.unique(patches[peaks]))      # patches with no peak keep one seed each
    seeds[unseeded] = patches[unseeded] + n
    seeds = seeds.astype(np.int32)
    seeds[~mask] = -1                                                   # background is filled last
    landscape = np.clip((distance.max() - distance) * 10, 0, 65535).astype(np.uint16)
    labelled = ndimage.watershed_ift(landscape, seeds)
    labelled[~mask] = 0
    return labelled


def buildings_from_labels(labelled):
    """One row per numbered building: pixel count, footprint and bounding box. Tiny ones are dropped."""
    rows = []
    for i, box in enumerate(ndimage.find_objects(labelled), start=1):
        if box is None:
            continue
        pixels = int((labelled[box] == i).sum())
        if pixels < MIN_BUILDING_PIXELS:
            labelled[box][labelled[box] == i] = 0
            continue
        rows.append({"building": i, "pixels": pixels, "footprint_m2": pixels * PIXEL_SIZE_M ** 2,
                     "y0": box[0].start, "y1": box[0].stop - 1, "x0": box[1].start, "x1": box[1].stop - 1})
    return pd.DataFrame(rows, columns=["building", "pixels", "footprint_m2", "y0", "y1", "x0", "x1"]), labelled


def find_buildings(finder, before, after, min_separation_px="default"):
    """Building map -> separate buildings. Returns a table with one row per building and the
    numbered map (0 = background, n = building n)."""
    separation = SPLIT_MIN_SEPARATION_PX if min_separation_px == "default" else min_separation_px
    return buildings_from_labels(split_touching(building_mask(finder, before, after), separation))


# -------------------------------------------------------------- step 2: judge damage
def crop_box(image, x0, x1, y0, y1, margin=CROP_MARGIN, size=CROP_SIZE):
    """Same arithmetic as the main notebook's crop_building(), given the building's extent."""
    height, width = image.shape[:2]
    xd, yd = x1 - x0, y1 - y0
    xmin, xmax = max(int(x0 - xd * margin), 0), min(int(x1 + xd * margin), width)
    ymin, ymax = max(int(y0 - yd * margin), 0), min(int(y1 + yd * margin), height)
    if xmax <= xmin or ymax <= ymin:
        return None
    return np.array(Image.fromarray(image[ymin:ymax, xmin:xmax, :3]).resize((size, size), Image.BILINEAR))


def judge_damage(damage_model, before, after, boxes):
    """boxes: table with x0, x1, y0, y1 per building. Returns a damage class per row."""
    pairs = []
    for b in boxes.itertuples():
        pre, post = crop_box(before, b.x0, b.x1, b.y0, b.y1), crop_box(after, b.x0, b.x1, b.y0, b.y1)
        pairs.append(np.zeros((CROP_SIZE, CROP_SIZE, 6), np.uint8) if pre is None else np.concatenate([pre, post], axis=-1))
    if not pairs:
        return []
    scores = damage_model.predict(np.stack(pairs).astype(np.float32), batch_size=256, verbose=0)
    return [DAMAGE_CLASSES[i] for i in scores.argmax(axis=1)]


# ------------------------------------------------- step 3: materials, carbon and cost
def _reference_tables():
    gem = pd.read_csv(REFERENCE_DIR / "gem_exposure_adm1.csv")
    rasmi = pd.read_csv(REFERENCE_DIR / "rasmi_material_intensity.csv")
    ice = pd.read_csv(REFERENCE_DIR / "ice_carbon_factors.csv")   # build from your own ICE download (see README)
    materials = sorted(m for m in rasmi["material"].unique() if m not in EXCLUDED_MATERIALS)
    return gem[gem["total_floor_area_m2"] > 0], rasmi, ice.set_index("material")["kgco2e_per_kg"], materials


def region_factors(iso3, gem_region, intensity_col="intensity_kg_per_m2"):
    """What 1 m2 of building FOOTPRINT means in one GEM region (Layer 1 regional mix):
    floor area, kg of each material, embodied carbon by both routes, replacement cost."""
    gem, rasmi, ice, materials = _reference_tables()
    c = gem[(gem["iso3"] == iso3) & (gem["adm1_name"] == gem_region)].copy()
    if c.empty:
        raise ValueError(f"no GEM data for {iso3} / {gem_region}")
    c["share"] = c["buildings"] / c["buildings"].sum()
    c["structure"] = c["macro_taxonomy"].map(GEM_TO_RASMI_STRUCTURE)
    c["function"] = np.where(c["occupancy"] != "RES", "NR", np.where(c["avg_storeys"] >= MULTI_FAMILY_MIN_STOREYS, "RM", "RS"))
    c["floor_m2"] = c["share"] * c["avg_storeys"]
    c = c.merge(rasmi[rasmi["material"].isin(materials) & (rasmi["region"] == COUNTRY_RASMI_REGION[iso3])],
                left_on=["structure", "function"], right_on=["structural_type", "functional_type"])
    one_class = c.drop_duplicates(["occupancy", "macro_taxonomy"])
    out = {"floor_m2": one_class["floor_m2"].sum(),
           "gem_kgco2e": (one_class["floor_m2"] * 1000 * one_class["embodied_carbon_tco2e"] / one_class["total_floor_area_m2"]).sum(),
           "cost_usd": (one_class["floor_m2"] * one_class["bldg_replacement_cost_usd"] / one_class["total_floor_area_m2"]).sum()}
    for m in materials:
        rows = c[c["material"] == m]
        out[f"kg_{m}"] = (rows["floor_m2"] * rows[intensity_col]).sum()
    out["ice_kgco2e"] = sum(out[f"kg_{m}"] * ice[m] for m in materials)
    return pd.Series(out)


def estimate(buildings, factors, damage_col="damage_class"):
    """Per-building rebuilt floor area, embodied material carbon (both routes) and replacement cost."""
    out = buildings.copy()
    out["rebuild_pct"] = out[damage_col].map(REBUILD_PCT)
    rebuilt_footprint = out["footprint_m2"] * out["rebuild_pct"]
    out["rebuilt_floor_m2"] = rebuilt_footprint * factors["floor_m2"]
    out["carbon_A_tco2e"] = rebuilt_footprint * factors["ice_kgco2e"] / 1000      # RASMI x ICE
    out["carbon_B_tco2e"] = rebuilt_footprint * factors["gem_kgco2e"] / 1000      # GEM
    out["replacement_cost_usd"] = rebuilt_footprint * factors["cost_usd"]         # GEM
    return out


def run(finder, damage_model, before, after, iso3, gem_region):
    """Whole pipeline on one image pair. Returns (per-building table, labelled building map)."""
    buildings, labelled = find_buildings(finder, before, after)
    buildings["damage_class"] = judge_damage(damage_model, before, after, buildings)
    return estimate(buildings, region_factors(iso3, gem_region)), labelled
