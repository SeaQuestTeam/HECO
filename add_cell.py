import json

with open('heco/sensivity_analysis/sensivity_analysis_2.ipynb', 'r') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from scipy.optimize import curve_fit\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "metrics_last_iteration = pd.read_csv('sa_2_2/heco_results_test_polygons_metrics_last_iteration.csv')\n",
        "\n",
        "def power_law(x, a, b):\n",
        "    return a * (x ** b)\n",
        "\n",
        "# plot diagrams of metrics_last_iteration\n",
        "#sort by perturbation distance\n",
        "metrics_last_iteration = metrics_last_iteration.sort_values(by=['origin_perturbation'], ascending=[True])\n",
        "\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.title('Metrics vs Perturbation distance for last LPDM iteration (with 90% Confidence Bounds)')\n",
        "plt.xlabel('Perturbation distance (m)')\n",
        "plt.ylabel('Metric value')\n",
        "\n",
        "variables = [\n",
        "    ('SRA', 'Success Rate Area (SRA)', 'tab:blue'),\n",
        "    ('CI', 'Centroid Index (CI)', 'tab:orange'),\n",
        "    ('Jaccard', 'Jaccard Similarity Index', 'tab:green'),\n",
        "    ('DICE', 'Dice Similarity Coefficient (DSC)', 'tab:red')\n",
        "]\n",
        "\n",
        "x_data = metrics_last_iteration['origin_perturbation'].values\n",
        "# avoid zero for power law\n",
        "x_fit = np.linspace(max(0.1, x_data.min()), x_data.max(), 100)\n",
        "\n",
        "for var, label, color in variables:\n",
        "    y_data = metrics_last_iteration[var].values\n",
        "    plt.scatter(x_data, y_data, label=label, color=color, alpha=0.6)\n",
        "    \n",
        "    # Fit power law trendline\n",
        "    try:\n",
        "        mask = (x_data > 0) & (y_data > 0)\n",
        "        if np.sum(mask) > 2:\n",
        "            popt, pcov = curve_fit(power_law, x_data[mask], y_data[mask], maxfev=10000)\n",
        "            plt.plot(x_fit, power_law(x_fit, *popt), color=color, linestyle='--')\n",
        "            \n",
        "            # Calculate 90% confidence bounds using Monte Carlo sampling\n",
        "            samples = np.random.multivariate_normal(popt, pcov, 1000)\n",
        "            y_fits = np.array([power_law(x_fit, *s) for s in samples])\n",
        "            lower_bound = np.percentile(y_fits, 5, axis=0)\n",
        "            upper_bound = np.percentile(y_fits, 95, axis=0)\n",
        "            plt.fill_between(x_fit, lower_bound, upper_bound, color=color, alpha=0.15)\n",
        "    except Exception as e:\n",
        "        print(f'Could not fit trendline for {var}: {e}')\n",
        "\n",
        "plt.legend()\n",
        "plt.ylim(0, 1)\n",
        "plt.grid()\n",
        "plt.savefig('sa_2_2/metrics_vs_origin_perturbation_last_iteration_bounds.png')\n",
        "plt.show()\n"
    ]
}

nb['cells'].append(new_cell)

with open('heco/sensivity_analysis/sensivity_analysis_2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Appended a new cell at the bottom of the notebook.")
