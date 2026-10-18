"""Assemble the public repository and its release files from the working project folder.

Only the files listed in PUBLIC_FILES are copied (an allow-list, so nothing else can leak).
Notebooks are cleaned on the way: warning messages (which contain local file paths) are
removed, and the output of the cell that displays the ICE carbon factors is cleared, because
ICE data may not be redistributed. A final scan refuses to finish if any ICE factor value or
local path is still present.

Usage (from the venv root):  bin/python xbd_project/tools/export_public_repo.py
Creates:  ENGG2112_XBD_Model/          the repository contents (git repo lives here)
          ENGG2112_release_assets/     zips to attach to a GitHub release
"""
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
REPO = PROJECT.parent / "ENGG2112_XBD_Model"
ASSETS = PROJECT.parent / "ENGG2112_release_assets"

PUBLIC_FILES = [
    "README.md", "METHODOLOGY.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "requirements.txt", ".gitignore",
    "xBD_damage_to_carbon_cost.ipynb", "prototype.ipynb", "xbd_pipeline.py",
    "colab_train.ipynb", "colab_train_multitask.ipynb",
    "tools/make_colab_notebook.py", "tools/build_ice_factors.py", "tools/export_public_repo.py",
    "reference_data/README.md", "reference_data/gem_exposure_adm1.csv", "reference_data/rasmi_material_intensity.csv",
]
ICE_DISPLAY_CELLS = {"92fbdfc5"}      # main notebook cell that shows the ICE factor table

# Final model weights, for the prototype
MODEL_FILES = ["p3_effnet_ce+tier3_cosine_20ep_150000spe.weights.h5", "mt_unet_effnet_30ep_500steps.weights.h5"]
# Cached results: everything in outputs/ the notebook reads, except bulky regenerable inputs
CACHE_EXCLUDE = ("crops_", "mt_tiles_", "laptop_set1_backup", ".partial")


def clean_notebook(src, dst):
    nb = json.load(open(src))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        outputs = [o for o in cell.get("outputs", []) if not (o.get("output_type") == "stream" and o.get("name") == "stderr")]
        if cell.get("id") in ICE_DISPLAY_CELLS:
            outputs = []
        cell["outputs"] = outputs
    json.dump(nb, open(dst, "w"), indent=1)


def forbidden_strings():
    """ICE factor values and local paths that must not appear in the published files.
    Returns (anywhere, in_text): full-precision ICE values are searched everywhere; the short,
    rounded forms only in code and prose (results tables and data can contain the same short
    numbers by coincidence, e.g. a model score)."""
    ice = pd.read_csv(PROJECT / "reference_data" / "ice_carbon_factors.csv")["kgco2e_per_kg"].dropna()
    forms = {repr(float(v)) for v in ice} | {f"{v:.6f}" for v in ice} | \
            {f"{v:.3f}".rstrip("0").rstrip(".") for v in ice} | {f"{v:.4f}".rstrip("0") for v in ice}
    digits = lambda f: len(f.replace("-", "").replace(".", "").lstrip("0"))
    full = {f for f in forms if digits(f) >= 5}          # specific enough to search everywhere
    short = {f for f in forms if digits(f) < 5}          # could occur by coincidence: code and prose only
    paths = {"/" + "Users" + "/", "/var" + "/folders/", "/private" + "/tmp/"}   # written split so this file does not match itself
    return full | paths, {s for s in short if len(s) >= 4}


def scan(path, anywhere, in_text, root):
    """Problems in one file. For notebooks, cell sources and markdown get the stricter check."""
    problems = []
    if path.suffix == ".ipynb":
        nb = json.load(open(path))
        for cell in nb["cells"]:
            source = "".join(cell["source"])
            outputs = json.dumps(cell.get("outputs", []))
            problems += [f"cell {cell.get('id')} source: {s!r}" for s in anywhere | in_text if s in source]
            problems += [f"cell {cell.get('id')} output: {s!r}" for s in anywhere if s in outputs]
    else:
        text = path.read_text(errors="ignore")
        strict = path.suffix in {".py", ".md", ".txt"}
        problems += [repr(s) for s in (anywhere | in_text if strict else anywhere) if s in text]
    return [f"{path.relative_to(root)}: contains {p}" for p in problems]


def main():
    # Build and scan in a temporary folder first; the repository folder is only touched if the scan passes.
    stage = Path(tempfile.mkdtemp(prefix="public_repo_"))
    for name in PUBLIC_FILES:
        src, dst = PROJECT / name, stage / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        clean_notebook(src, dst) if name.endswith(".ipynb") else shutil.copy2(src, dst)

    anywhere, in_text = forbidden_strings()
    problems = []
    for path in stage.rglob("*"):
        if path.is_file() and path.suffix in {".ipynb", ".py", ".md", ".csv", ".txt"}:
            problems += scan(path, anywhere, in_text, stage)
    if problems:
        shutil.rmtree(stage)
        raise SystemExit("NOT SAFE TO PUBLISH (repository folder left unchanged):\n  " + "\n  ".join(problems))

    REPO.mkdir(exist_ok=True)
    for item in REPO.iterdir():                  # replace the contents but keep the git history
        if item.name != ".git":
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    for item in stage.iterdir():
        shutil.move(str(item), REPO / item.name)
    shutil.rmtree(stage)

    ASSETS.mkdir(exist_ok=True)
    out = PROJECT / "outputs"
    with zipfile.ZipFile(ASSETS / "trained_models.zip", "w", zipfile.ZIP_STORED) as z:
        for name in MODEL_FILES:
            z.write(out / name, name)
    with zipfile.ZipFile(ASSETS / "cached_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            rel = path.relative_to(out)
            if path.is_file() and not any(str(rel).startswith(x) or x in str(rel) for x in CACHE_EXCLUDE) and path.name != ".DS_Store":
                z.write(path, str(rel))

    print("Repository contents:", REPO)
    for path in sorted(p for p in REPO.rglob("*") if p.is_file() and ".git" not in p.parts):
        print(f"  {path.relative_to(REPO)}  ({path.stat().st_size / 1e6:.2f} MB)")
    print("Release files:", ASSETS)
    for path in sorted(ASSETS.iterdir()):
        print(f"  {path.name}  ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
