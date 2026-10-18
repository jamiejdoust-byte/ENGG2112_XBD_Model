# Reference data

| File | Source | Licence | How it was made |
|---|---|---|---|
| `gem_exposure_adm1.csv` | GEM Global Exposure Model v2026.0.0, Yepes-Estrada, C. et al. (2026), doi:10.5281/zenodo.8117363; publication: Yepes-Estrada et al. (2023), *Global Building Exposure Model for Earthquake Risk Assessment*, Earthquake Spectra, doi:10.1177/87552930231194048 | CC BY-NC-SA 4.0 (this derived file is shared under the same licence) | `World/summaries/Exposure_Summary_Adm1_Taxonomy.csv` filtered to the 9 countries of the xBD disasters, summed by state x occupancy x macro construction class; `avg_storeys` parsed from the taxonomy height codes (single value as is, range = midpoint, ">N" = N+1) |
| `rasmi_material_intensity.csv` | RASMI, Fishman, T. et al. (2024), doi:10.5281/zenodo.10782341 (`MI_ranges_20230905.xlsx`) | CC BY 4.0 | All eight material sheets stacked into one table; `intensity_kg_per_m2` is the median (p50), with p25 and p75 kept |
| `ice_carbon_factors.csv` | **not included**: ICE Database Educational may not be redistributed | educational use only | Build it from your own download with `python tools/build_ice_factors.py` |
