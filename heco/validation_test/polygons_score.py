
"""
Oil Spill Dispersion Metrics  – 
=============================================================
Computes three evaluation metrics comparing an observed SAR-detected oil slick
polygon (O) against a simulated dispersion polygon (S):
 
  1. Success Rate Area  (SRA) = Area(O ∩ S) / Area(O)
  2. Centroid displacement Index (CI)  = Δx / L_box
  3. Jaccard Similarity Index (JSI) = Area(O ∩ S) / Area(O ∪ S)
  4. DICE Similarity Coefficient (DSC) = 2 x Area(O ∩ S) / Area(O) + Area(S)
 
Requirements:
    pip install geopandas shapely pyproj
 
Usage:
    python oil_spill_metrics_geopandas.py
    # or import and call compute_metrics() directly
"""

import math
from pathlib import Path
 
import geopandas as gpd
from shapely.geometry import box

# ── File paths ────────────────────────────────────────────────────────────────
 
OBS_PATH = Path("results/qgis_comparison/30-AUG-2021-AbouSamra-and-Ai-2024-oilspill-SAR-detection.geojson")
SIM_PATH = Path("results/qgis_comparison/convex-hull-polygon-TEST-2-2021-08-30_10-36.geojson")
 
# UTM zone 36N  (EPSG:32636) — appropriate for the Eastern Mediterranean / Cyprus area

PROJECTED_CRS = "EPSG:32636"

def compute_metrics(obs_path: Path, sim_path: Path) -> dict:
    """
    Load two GeoJSON polygon files and compute SRA, CI, and Jaccard extension.
 
    Parameters
    ----------
    obs_path : Path
        GeoJSON file of the *observed* oil slick (O).
    sim_path : Path
        GeoJSON file of the *simulated* dispersion result (S).
 
    Returns
    -------
    dict with keys: SRA, CI, J, and several intermediate diagnostic values.
    """
 
    # ── 1. Load GeoDataFrames (geographic CRS, WGS-84) ───────────────────────
    gdf_O = gpd.read_file(obs_path)   # observed  – SAR detection
    gdf_S = gpd.read_file(sim_path)   # simulated – convex-hull dispersion
 
    # ── 2. Reproject to a metric CRS (metres) ────────────────────────────────
    gdf_O = gdf_O.to_crs(PROJECTED_CRS)
    gdf_S = gdf_S.to_crs(PROJECTED_CRS)
 
    # Take the first feature of each layer
    geom_O = gdf_O.geometry.iloc[0]
    geom_S = gdf_S.geometry.iloc[0]
 
    # ── 3. Set operations (shapely) ───────────────────────────────────────────
    geom_intersection = geom_O.intersection(geom_S)
    geom_union        = geom_O.union(geom_S)
 
    area_O     = geom_O.area              # m²
    area_S     = geom_S.area              # m²
    area_inter = geom_intersection.area   # m²
    area_union = geom_union.area          # m²
 
    # ── 4. Centroids ──────────────────────────────────────────────────────────
    centroid_O = geom_O.centroid          # shapely Point (x_m, y_m)
    centroid_S = geom_S.centroid
 
    delta_x = centroid_O.distance(centroid_S)   # metres (Euclidean in proj. CRS)
 
    # ── 5. Bounding-box diagonal of O (L_box) ────────────────────────────────
    minx, miny, maxx, maxy = geom_O.bounds
    L_box = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)   # metres
 
    # ── 6. Metrics ────────────────────────────────────────────────────────────
    SRA = area_inter / area_O     if area_O     > 0 else float("nan")
    CI  = delta_x   / L_box      if L_box      > 0 else float("nan")
    J   = area_inter /area_union if area_union > 0 else float("nan")
    DICE = (2*area_inter) / (area_S + area_O) if (area_O+area_S)>0 else float("nan")
 
    return {
        # Primary metrics
        "SRA": SRA,
        "CI":  CI,
        "J":   J,
        "DICE": DICE,
        # Diagnostic / intermediate values
        "area_O_km2":     area_O     / 1e6,
        "area_S_km2":     area_S     / 1e6,
        "area_inter_km2": area_inter / 1e6,
        "area_union_km2": area_union / 1e6,
        "centroid_O":     (centroid_O.x, centroid_O.y),
        "centroid_S":     (centroid_S.x, centroid_S.y),
        "delta_x_km":     delta_x / 1e3,
        "L_box_km":       L_box   / 1e3,
    }
 
 
# ── Pretty printer ────────────────────────────────────────────────────────────
 
def print_report(m: dict) -> None:
    SEP = "─" * 62
    print(SEP)
    print("  OIL SPILL DISPERSION METRICS")
    print(SEP)
    print(f"  Area O  (observed)           : {m['area_O_km2']:>10.1f}  km²")
    print(f"  Area S  (simulated)          : {m['area_S_km2']:>10.1f}  km²")
    print(f"  Area (O ∩ S)  intersection   : {m['area_inter_km2']:>10.1f}  km²")
    print(f"  Area (O ∪ S)  union          : {m['area_union_km2']:>10.1f}  km²")
    print(f"  Centroid distance Δx         : {m['delta_x_km']:>10.1f}  km")
    print(f"  Bounding-box diagonal L_box  : {m['L_box_km']:>10.1f}  km")
    print(SEP)
    print(f"  Success Rate Area      (SRA) : {m['SRA']:>10.4f}   [0–1,  higher ↑ = better]")
    print(f"  Centroid Index          (CI) : {m['CI']:>10.4f}   [≥ 0,  lower  ↓ = better]")
    print(f"  Jaccard Similarity Index (J) : {m['J']:>10.4f}   [0–1,  higher ↑ = better]")
    print(f"  Dice Similarity Coeff. (DSC) : {m['DICE']:>10.4f}   [0–1,  higher ↑ = better]")
    print(SEP)
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    metrics = compute_metrics(OBS_PATH, SIM_PATH)
    print_report(metrics)