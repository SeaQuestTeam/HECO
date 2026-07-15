import importlib.util
from pathlib import Path

import geopandas as gpd
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "heco" / "sensivity_analysis" / "heco.py"


def load_module():
    spec = importlib.util.spec_from_file_location("heco_sens", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_open_yaml_falls_back_to_local_dataset_when_credential_path_is_missing(tmp_path):
    heco = load_module()
    dataset_path = ROOT / "heco" / "HECO_TEST.nc"
    yaml_path = tmp_path / "config.yaml"

    yaml_path.write_text(
        yaml.safe_dump(
            {
                "input": {
                    "dataset_file_name": str(dataset_path),
                    "lat0": 40.5,
                    "lon0": 10.5,
                    "sim_diffusion_coeff": 10.0,
                    "sim_duration_h": 72.0,
                    "sim_particles": 100.0,
                    "sim_timedelta_s": 3600.0,
                    "spill_release_duration_h": 6.0,
                    "time0": "2025-03-08 00:00:00",
                    "volume_spilled_m3": 1000.0,
                }
            }
        )
    )

    inputdata, ds = heco.open_yaml(str(yaml_path))

    assert inputdata["dataset_file_name"] == str(dataset_path)
    assert ds is not None


def test_create_webmap_handles_single_point_geojson(tmp_path):
    heco = load_module()
    points = gpd.GeoDataFrame(
        {"time": ["2025-03-08T00:00:00"], "value": [1]},
        geometry=gpd.points_from_xy([10.5], [40.5]),
        crs="EPSG:4326",
    )
    geojson_path = tmp_path / "points.geojson"
    points.to_file(geojson_path, driver="GeoJSON")

    output_path = tmp_path / "out" / "map.html"
    heco.create_webmap(
        str(geojson_path),
        EMODnetLayers=False,
        settingsFile_path=str(ROOT / "heco" / "sensivity_analysis" / "operational_test_1_2.yaml"),
        output_path=str(output_path),
        savepolygons=False,
    )

    assert output_path.exists()
