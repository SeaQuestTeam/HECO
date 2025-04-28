![img](markdown_assets/HECO-4.png)

# HEre Comes the Oil

*HECO* (HEere Comes the Oil) is an advanced system for monitoring and forecasting oil dispersion at sea, designed to ensure a rapid response to accidental oil spills. Its main innovation lies in the integration of a high-performance computational model with real-time maritime and environmental data, enabling a rapid and accurate assessment of the impact of pollution.

>[!note]
>A live example of web map result is available here [heco/heco_map.html](https://seaquestteam.github.io/HECO/heco/heco_map.html)

### Table of Contents:

1. [Installation](#1-installation)
2. [Workflow](#2-workflow)

    2.1 [Fast workflow without Jupyyter notebook](#21-fast-workflow)
3. [Contributing](#3-contributing)
4. [Licence](#4-license)

---

>[!Warning]
> The simuator algorytm is not validated by peer reviewers.
> This is a Proof of Concept product! USE IT FOR TESTING PURPOSES ONLY!

This project uses the Copernicus Marine Services user credentials (username and password) to access the CMSES dataset API and retrieve ocean current wave variables.

More info about `copernicusmarine` API --> [link](https://help.marine.copernicus.eu/en/articles/8287609-copernicus-marine-toolbox-api-open-a-dataset-or-read-a-dataframe-remotely)

## 1. Installation

>[!Note]
> A very easy way to run HECO is to clone this repo inside of an EDITO data lab!
>
> 1. Go to [https://datalab.dive.edito.eu](https://datalab.dive.edito.eu) and sing-up or login
>
> 2. Go to "Service Catalog" and launch a "Jupyter-python-ocean-science" with default configuration (ignore warnings)
>
> 3. Start the jupyter server with access token gived at launch (copy and past the token password)
>
> 4. From the Git menu (look for the Git icon on the left), click on 'Clone Repository'.
>
> 5. Paste the url of this repo and clone, then run the console command `pip install -r requirements.txt`
>
> 6. Go to notebook `heco/HECO.ipynb` and follow instructions.

---
If you prefer install on your local machine follow these step

To set up the project, clone the repository and install the required dependencies. You can do this by running the following commands from the CLI:

```
virtualenv heco 

source bin/activate # Linux/Mac
Scripts\activate     # Windows

pip install -r requirements.txt
```

The file `requirements.txt` contains a list of all package and python dependencies required (and others useful too). The file `heco.py` contains all the functions developed for this tool.

## 2. Workflow

Follow the instruction in the computational notebook [HECO](heco/HECO.ipynb)

The procedure is divided into two main steps.

In the first step, an oil spill scenario is calculated using the wave velocity forecast data set in a dispersion-lagrangian model.

![gif](markdown_assets/scatter.gif)

A second step is to generate geo-spatial features to assess the impact on human activities and natural protected areas. Using a powerful Python script, HECO produces a one-page web map with an animation of the spill and some geodata in a matter of seconds. Useful for web sharing and early warning communications.


![hecomap](markdown_assets/heco_map_LD.gif)

### 2.1 a "Fast track" without Juputer notebook (in 10 STEPS)

A more fast way to use HECO in 10 STEPS is described below:
>
>1. Go to Data Access of MEDSEA_ANALYSISFORECAST_PHY - [data.marine.copernicus.eu](https://data.marine.copernicus.eu/product/MEDSEA_ANALYSISFORECAST_PHY_006_013/description)
>2. Go to the product `cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m` and click to ["SUBSET->Form"](https://data.marine.copernicus.eu/product/MEDSEA_ANALYSISFORECAST_PHY_006_013/download?dataset=cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m_202411)
>3. Use the map gui to draw a box for you ROI
>4. chose Start and End Date (note: forecasting are available for next 8 days)
>5. click "Download" and retrive the `.nc` file
>6. move the file in the project folder (or upload in cloud computing environment storage)
>7. Compile the settings input yaml file, using as template [heco.yaml](heco/heco.yaml), respect the identation and manual entry these values:
>
> - `dataset_file_name:` insert the local path of the `.nc` file (downloaded before)
> - `lat0:` insert the latitude of oilspill origin (WGS84)
> - `lat0:` insert the longitude of oilspill origin (WGS84)
> - `sim_diffusion_coeff:` Diffusion coefficent (min 1, max 100), default 10
> - `sim_duration_h:` insert the forecasting simulation in hours
> - `sim_particles:` the number of lagrangian particles to consider, this value has an impact on computational resources needed
> - `sim_timedelta_s:` default 3600 (1h), i. this the step in seconds for each iteration (the same as the .nc dataset timedelta)
> - `spill_release_duration_h:` Parameter to perform a discrete calculation for a continued spill scenario. a value >1 will perform a distribution of the spilled volume into multiple single instantaneous spills.
> - `time0:` insert the time of oilspill event in format `2025-03-08 00:00:00
> - `volume_spilled_m3:` insert the estimate volume spilled in entire release duration.
>
> 8. Open a python terminal and `import heco`
> 9. Run the model simulation: `output = heco.run('heco.yaml')`
> 10. Save output in tabular CSV `output.to_csv('heco_results.csv', index=False)`

The tabular CSV contain the RAW particles data, it is possibile to use it in any GIS environment.

## 3. Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## 4. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.