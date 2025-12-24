# Hydrodynamic-Amplification-of-Weak-Symmetry-Breaking-in-Active-Granular-Fluids
Resolving the "directional selection puzzle," this study uses GPU simulations of 12,000 particles to identify hydrodynamic amplification. It demonstrates that at intermediate densities, fluid coupling amplifies weak biases by ~17x, triggering a sharp phase transition (\beta_c \approx 0.92) into locked vortex 

Copyright (c) 2025 Mohamed Hosam Mohamed Aly Zahran




**Author:** Mohamed Hosam Mohamed Aly Zahran  
**Affiliation:** Independent Computational Physics Research, Cairo, Egypt / Menoufia National University  
**Correspondence:** mohamed.zahran.0315@med.mnu.edu.eg  
**Date:** December 2025

---

## 📌 Abstract
This repository contains the source code and datasets for the research paper **"Hydrodynamic Amplification of Weak Symmetry Breaking in Active Granular Fluids."**

We investigate the emergence of macroscopic vortex formation in active granular fluids using large-scale GPU-accelerated simulations ($N = 12,000$). The study identifies a critical bias threshold $\beta_c = 0.920 \pm 0.032$ at which weak symmetry breaking triggers a sharp phase transition from chaotic motion to locked vortex states. This mechanism achieves a **~17× amplification factor**, operating exclusively within an intermediate density regime ($0.35 < \phi < 0.65$).

## 🚀 Key Findings
1.  **Critical Phase Transition:** A sharp transition occurs at $\beta_c \approx 0.92$ with critical-like steepness ($\kappa \approx 14.3$).
2.  **Hydrodynamic Amplification:** Weak external biases are amplified by a factor of 17x due to collective fluid coupling.
3.  **Density Dependence:** The mechanism works only in the "fluid" regime; gaseous and jammed phases show negligible amplification.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `simulation_core.py` | Main JAX script for running the GPU-accelerated active matter simulation. |
| `cold_flow_results.csv` | Raw data containing the vortex order parameter ($\Phi$) vs. time. |
| `ramp_results.csv` | Data from the bias ramp protocol used to identify $\beta_c$. |
| `analysis_plots.ipynb` | Jupyter Notebook used to generate the sigmoid fit and figures. |
| `Paper.pdf` | The full text of the research paper. |

## 💻 Installation & Usage

This project relies on **JAX** for high-performance GPU computing.

### 1. Prerequisites
You need Python 3.9+ and the following libraries:
```bash
pip install --upgrade "jax[cuda12_pip]" -f [https://storage.googleapis.com/jax-releases/jax_cuda_releases.html](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html)
pip install numpy matplotlib pandas scipy

(Note: Adjust the JAX installation command based on your CUDA version).
2. Running the Simulation
To run the "Cold Flow" protocol with default parameters:
python simulation_core.py --N 12000 --steps 50000 --density 0.50

📊 Data Format
The ramp_results.csv file contains the following columns:
 * step: Simulation timestep.
 * bias_strength: The applied rotational bias (\beta).
 * order_parameter: The measured vortex order (\Phi).
 * density: Packing fraction (\phi).
🤖 AI Assistance Declaration
The mathematical models, force equations, and numerical integration schemes in this research were developed with assistance from Artificial Intelligence tools (Gemini). The AI assisted in designing the JAX-based GPU implementation and optimizing force calculations. All scientific interpretations and final parameters were determined by the author.
📜 Citation
If you use this code or data in your research, please cite:
> Zahran, M. H. M. A. (2025). Hydrodynamic Amplification of Weak Symmetry Breaking in Active Granular Fluids. GitHub Repository. https://github.com/[YOUR-USERNAME]/[REPO-NAME]
> 
📄 License
 * Code: MIT License
 * Data & Text: CC-BY 4.0 International
