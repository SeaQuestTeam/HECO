# 🌊 HECO: HEre Comes the Oil

**HECO** is an advanced system for monitoring and forecasting oil dispersion at sea. It integrates high-performance computational models with real-time environmental data to provide rapid, accurate impact assessments for accidental spills.

> [!CAUTION]
> **Proof of Concept:** The simulation algorithm is not peer-reviewed. This tool is for **testing and research purposes only.**

🔗 **Live Demo:** [View Interactive Web Map](https://seaquestteam.github.io/HECO/heco/heco_map.html)

---

## 🚀 1. Quick Start (EDITO Data Lab)

The easiest way to run HECO is within the **EDITO** cloud environment.

1. Log in to [datalab.dive.edito.eu](https://datalab.dive.edito.eu).
2. In the "Service Catalog," launch **Jupyter-python-ocean-science** (default settings).
3. Open the Jupyter server using the access token provided at launch.
4. Use the **Git** menu (left sidebar) to `Clone Repository` using this repo's URL.
5. Open the terminal and install dependencies:
```bash
pip install -r requirements.txt

```


6. Navigate to `heco/HECO.ipynb` and follow the notebook instructions.

### Local Installation

To run HECO on your machine, clone the repo and set up your environment:

```bash
virtualenv heco 
source heco/bin/activate  # Linux/Mac
# heco\Scripts\activate   # Windows
pip install -r requirements.txt

```

---

## 🛠️ 2. Workflow

The procedure consists of two main phases:

1. **Simulation:** Calculating the oil spill scenario using a Lagrangian dispersion model powered by Copernicus (CMEMS) wave velocity forecasts.
2. **Geoprocessing:** Automatically generating a one-page animated web map to assess impacts on human activities and protected natural areas.

---

## ⚡ 3. Fast Track (CLI / Scripting)

You can bypass the Jupyter notebook and run the model directly via Python using a configuration file.

### A. Configuration

Create a `heco.yaml` file based on this template:

| Variable | Unit | Description |
| --- | --- | --- |
| `credential_path` | str | Path to YAML file with Copernicus credentials |
| `lat0` / `lon0` | deg | Latitude/Longitude of the spill origin |
| `sim_diffusion_coeff` | int | Diffusion coefficient (1-100, default: 10) |
| `sim_duration_h` | hours | Forecasting simulation length |
| `sim_particles` | int | Number of Lagrangian particles (impacts performance) |
| `time0` | date | Event timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `sim_timedelta_s:`|int| time step in seconds, default 3600 (1h), for each iteration (must be the same as the dataset timeseries)|
|`spill_release_duration_h:`|int| Parameter to perform a discrete calculation for a continued spill scenario. a value >1 will perform a distribution of the spilled volume into multiple single instantaneous spills|
|`volume_spilled_m3:`|float| insert the estimated volume spilled in entire release duration|
| `dataset_file_name`* | str | (Optional) Path to a local `.nc` file |

> [!TIP]
> **Performance Hack:** For faster computation, download the `.nc` marine currents forecast dataset manually from [Copernicus Marine](https://data.marine.copernicus.eu/product/MEDSEA_ANALYSISFORECAST_PHY_006_013/) and use the `dataset_file_name` parameter instead of API credentials.

### B. Execution

Run the simulation in your Python environment:

```python
import heco

# 1. Run the model simulation
output = heco.run('heco.yaml')

# 2. Export raw particle data for GIS
output.to_csv('heco_results.csv', index=False)

```

---

## 🤝 Contributing & License

* **Contributing:** Please fork the repository and submit a pull request with your changes.
* **License:** This project is licensed under the **MIT License**: Copyright (c) 2025 Gianfranco Di Pietro, Massimiliano Marino, Martina Stagnitti, Sofia Nasca, Elisa Castro, Rosaria Ester Musumeci - University of Catania - Department of Civil and Architecture Engineering.
