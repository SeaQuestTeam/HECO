# HECO: HEre Comes the Oil

HECO simulates oil-spill particle trajectories from ocean-current or Stokes-drift fields and exports point tracks, convex hulls, and web maps. This repository contains the code, input settings, selected results, and notebooks used for the analyses reported in the accompanying manuscript.

## Set up a local environment

The notebooks were prepared with Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python restore_data.py
```

`restore_data.py` checks the SHA-256 digest of the split `HECO_TEST.nc` archive and places the dataset in the two directories used by the demonstration and sensitivity notebooks. The two validation forcing files are already under `heco/validation_test/`. The bundled data can be used without a Copernicus Marine login. For a new online download, keep credentials outside this repository and pass their location through the configuration file.

Open each notebook with its own directory as the Jupyter working directory. For example:

```bash
cd heco/sensivity_analysis
jupyter lab
```

The calibration and replicate simulation cells can take substantial time. Their existing notebook outputs and the corresponding result tables are included so the reported plots can be inspected before rerunning the experiments.

## Files linked to the manuscript

| Analysis | Notebook | Inputs and saved results |
| --- | --- | --- |
| HECO demonstration | `heco/HECO.ipynb` | `heco/heco_test.yaml`, `HECO_TEST.nc`, point results and web map in `heco/` |
| Spill-origin perturbation | `heco/sensivity_analysis/sensivity_analysis_3.ipynb`; `perturbation-origin-assessment.ipynb`; `sensitivity_visualizations.ipynb` | `sa_2_3/` contains 300 settings, 300 convex-hull files, and the metric tables; `figures_sensitivity/` contains the four metric maps used in the article |
| Spilled-volume sensitivity | `heco/sensivity_analysis/sensivity_analysis_5_paired_replicates.ipynb` | `sa_2_5_common_seed_replicates/` contains five common-seed replicates, metric tables, and the published plot |
| Baniyas hindcast | `heco/validation_test/HECO-validation-test.ipynb` (Test 1, Stokes drift); `HECO-validation-test-2.ipynb` (Test 2, currents) | Two 52-origin configuration sets, local forcing files, saved results, observed polygons, and the validation map |
| Diffusion calibration | `heco/validation_test/HECO-validation-test-3_calibration_diffusion_parallel.ipynb` | `calibration_D_test/` contains the 150-coefficient score table, final hulls, rankings, and the published calibration plot |

The `heco.py` copies in the demonstration, sensitivity, and validation directories are identical. The notebooks import the local module from their working directory. Each analysis also has its own `polygons_score.py` because the scoring implementations differ. `paper_figures/` holds the composite validation image used in the manuscript.

## Scope and interpretation

The origin analysis reports 298 usable perturbation comparisons from the saved metric table. The volume experiment uses five seeds and 25 volume settings per seed. In that experiment, volume changes the particle count while volume per particle remains fixed; the convex-hull response is therefore a particle-sampling proxy, not a physical model of slick thickness or weathering.

Generated calibration input YAML files and the full point trajectories from the origin experiment are omitted because their notebooks regenerate them from the included settings and forcing data. The saved metric tables, convex hulls, validation results, and notebook outputs remain available for inspection. Preliminary experiments and the local Python environment are outside this repository.

The origin and validation notebooks do not store the random generator state used for their original particle tracks. A fresh simulation follows the same settings but may not reproduce the saved trajectories byte for byte. Web-map layers and background map tiles may also require network access when figures are regenerated.

## License

See [LICENSE](LICENSE).
