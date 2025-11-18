# 3D Vectorized MOT Simulation Project Workflow

This project implements a **fully vectorized simulation** of a Magneto-Optical Trap (MOT) using the full kinetic model, designed for computational efficiency on high-performance clusters. The workflow is optimized for retrieving and analyzing large simulation datasets locally.

---

## 🔬 Project Files

| **File** | **Role** | **Description** |
| :--- | :--- | :--- |
| **`mot_simulation_full.py`** | **Simulation** (Run Remotely) | Runs the $N$-body simulation (Doppler, Zeeman, Attenuation). Saves output to `mot_data_full.npz`. |
| **`mot_post_processor_improved.py`** | **Post-Processing** (Run Locally) | Loads data, generates **3 verification figures**, and creates the **evolution GIF**. Includes the crucial **SCP utility** for automated data retrieval. |

---

## 🧪 Simulation Details: Low-Density Mode

The current simulation parameters in **`mot_simulation_full.py`** are configured for the **Low-Density (Spherical) MOT** regime. This allows observation of the ideal cooling and confinement phase without being dominated by collective light attenuation.

* **Atom Number ($N$):** 1,000
* **Initial Velocity:** Very High ($\mathbf{5.0\ \text{m/s}}$ RMS)
* **Expected Result:** Rapid cooling followed by stabilization into a small, nearly **spherical** atom cloud, matching typical laboratory observations of non-dense MOTs.

---

## 🚀 Step-by-Step Workflow Guide

### Step 1: Run the Simulation (Remote Cluster)

Execute the simulation script on your cluster environment. This will create the data file defined in your configuration.

# Execute remotely (e.g., via a SLURM job)

```python
**`python mot_simulation_full.py`**
```

### Step 2: Local Setup and Configuration (Crucial Step)

Before analysis, you must configure your access details within the post-processor script.

Action Required: Open mot_post_processor_improved.py and edit the variables in the >>> USER-DEFINED CLUSTER SETTINGS <<< block:
Python

# USER-DEFINED CLUSTER SETTINGS
- CLUSTER_USER = "your_username"        # e.g., "leona"
- CLUSTER_HOST = "cluster.server.edu"   # e.g., "login.cluster.uni"
- REMOTE_FILE_PATH = "/path/to/simulation/results/mot_data_full.npz" 

### Step 3: Data Retrieval and Post-Processing

You have two options for running the post-processor:

- **Option A**: Automated Download (Recommended)

If the data file (**`mot_data_full.npz`**) is not present locally, run the script with the download argument. This triggers the Python-based SCP transfer using your configured variables. It downloads data and starts analysis immediately

```python
python mot_post_processor_improved.py download
```

(Requires: SSH keys to be configured for password-less access or manual password input.)

- **Option B**: Local Analysis Only

If you manually copied the data file already, just run the script normally.

```python
python mot_post_processor_improved.py
```

Analyzes data already present in the local directory.

### 🛠️ Setup, Dependencies, and Verification

This project requires the following Python libraries for post-processing:

```bash
pip install numpy matplotlib pillow`
```

Output File	Key Verification Check
`Figure 1`	Stability and Cooling Rate
`Figure 2`	Final Cloud Shape (expected to be spherical for the default low-OD parameters)
`Figure 3`	Phase-Space Damping and Final Temperature
`mot_evolution.gif`	Visual confirmation of cloud shrinking