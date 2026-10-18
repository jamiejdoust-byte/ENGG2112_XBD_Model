"""Build reference_data/ice_carbon_factors.csv from YOUR OWN copy of the ICE Database.

The ICE (Inventory of Carbon & Energy) Educational database may not be redistributed
(Circular Ecology terms of use, clauses 16.12 and 15.4), so this repository does not
contain any ICE values. Download it yourself and run this script:

  1. Download "ICE Database Educational V5.0" from
     http://www.circularecology.com/ice-database.html  (free, for educational use only)
  2. Unzip it so the .xlsx file is in reference_data/ice_raw/
  3. From the project folder:  python tools/build_ice_factors.py

The script reads one factor per material from the sheet "ICE Summary", checks that each
row has the expected name (so a different ICE version is caught rather than silently
read wrong), and writes the CSV the notebook and xbd_pipeline.py expect.
Please give full attribution to the ICE Database and Circular Ecology Ltd. when using it.
"""
import glob
import sys
from pathlib import Path

import openpyxl
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
SHEET = "ICE Summary"
NAME_COLUMN, VALUE_COLUMN = 6, 7          # 1-based spreadsheet columns F (material name) and G (kgCO2e/kg)

# material, summary-sheet row, name expected in column F, which ICE entry it is, why this one
CHOICES = [
    ("concrete", 279, "General", "Concrete > In-Situ Concrete > General",
     "ICE advises a specific mix where known; the general average is used because the mix is unknown"),
    ("brick", 123, "General (Common Brick)", "Bricks > General (Common Brick)", ""),
    ("wood", 928, "Timber - Average of all data - No Carbon Storage", "Timber > Average of all data - No Carbon Storage",
     "Excludes carbon stored in the timber (project decision; the including-storage factor is a sensitivity case)"),
    ("steel", 894, "Steel, Rebar", "Steel > worldsteel World Average > Steel, Rebar", "World average rebar"),
    ("glass", 620, "Glass, General, per kg", "Glass > Glass, General, per kg", ""),
    ("plastics", 827, "General", "Plastics > General", "Average of a wide range of plastic types"),
    ("aluminum", 49, "Aluminium General, Worldwide", "Aluminium > Worldwide production > Aluminium General, Worldwide",
     "World average recycled content"),
    ("wood_with_carbon_storage", 956, "Timber - Average of all data - Including Carbon Storage",
     "Timber > Average of all data - Including Carbon Storage", "Used only in the sensitivity analysis (Section 9b)"),
]


def main(xlsx_path=None):
    if xlsx_path is None:
        found = glob.glob(str(PROJECT / "reference_data" / "ice_raw" / "*.xlsx"))
        if not found:
            sys.exit("No ICE .xlsx found in reference_data/ice_raw/ - see the instructions at the top of this file.")
        xlsx_path = found[0]
    sheet = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)[SHEET]
    rows = {r: row for r, row in enumerate(sheet.iter_rows(min_row=1, max_row=1100, values_only=True), start=1)}
    out = []
    for material, row, expected_name, entry, note in CHOICES:
        name, value = rows[row][NAME_COLUMN - 1], rows[row][VALUE_COLUMN - 1]
        if str(name).strip() != expected_name:
            sys.exit(f"Row {row} of '{SHEET}' is '{name}', expected '{expected_name}'. "
                     "This script was written for ICE Educational V5.0 (Jun 2026); check the row numbers for your version.")
        out.append({"material": material, "kgco2e_per_kg": float(value), "ice_entry": entry, "ice_db_name": expected_name,
                    "ice_summary_row": row, "note": note})
    out.append({"material": "copper", "kgco2e_per_kg": None, "ice_entry": "Copper", "ice_db_name": "", "ice_summary_row": 611,
                "note": "Not available: ICE V5.0 withdrew its copper values. Copper is excluded from the pipeline."})
    table = pd.DataFrame(out)
    table["source"] = f"ICE Database Educational ({Path(xlsx_path).name}), sheet '{SHEET}'"
    target = PROJECT / "reference_data" / "ice_carbon_factors.csv"
    table.to_csv(target, index=False)
    print(f"Wrote {target} ({len(table)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
