# Human Gloss Perception and Tiny Neural Networks: Figure Generation and Data Processing

![Repository Thumbnail](thumbnail.jpg)

This repository contains the source code, example data, and scripts required to process behavioural and computational model data, and to generate all main and supplementary figures for the manuscript:

Takuma Morimoto, Arash Akbarinia, Katherine Storrs, Jacob R. Cheeseman, Hannah E. Smithson, Karl R. Gegenfurtner, and Roland W. Fleming
"Human gloss perception reproduced by tiny neural networks."
*bioRxiv*
[https://doi.org/10.1101/2025.05.09.653112](https://doi.org/10.1101/2025.05.09.653112)

---

## 1. System Requirements

* **Operating System:**
  Windows, macOS, or Ubuntu (tested on macOS ver 15.3.2, Ubuntu 24.04 LTS)

* **MATLAB Version:**
  R2020a or newer (tested on R2020a, R2023a, R2024b)

* **Required MATLAB Toolboxes:**

  * Image Processing Toolbox (v11.1, v24.2)
  * Statistics and Machine Learning Toolbox (v11.7, v24.2)
  * Curve Fitting Toolbox (v3.5.11, v24.2)
  * Computer Vision Toolbox (v9.2, v24.2)
  * Optimization Toolbox (v8.5, v24.2)

* **Hardware:**
  Standard desktop/laptop computer (no special hardware required)

---

## 2. Installation Guide

1. Download the repository:
   [https://github.com/takuma929/gloss\_tinynetworks](https://github.com/takuma929/gloss_tinynetworks)
   or clone with git:

   ```
   git clone https://github.com/takuma929/gloss_tinynetworks.git
   ```

2. Open MATLAB and set the repository root directory as your working directory.

3. Add all subfolders to the MATLAB path.

---

## 3. Demo: Generate All Figures

To generate all figures and process the data as in the manuscript:

1. Start MATLAB and navigate to the main project folder.

2. Add the main project folder and its subfolders to the MATLAB path.

3. Run the following command in the MATLAB Command Window:

   ```
   main
   ```

4. All figures will be saved to the `figs/` directory, and cleaned data files will be saved in the `data/` directory.

**Expected Output:**

* PDF and PNG files for all main and supplementary figures
* Cleaned MATLAB data files (e.g., `onlineData.mat`)

**Typical install & run times:**

* Setup: less than 1 minute (copy/unzip files and MATLAB path setup)
* Data download: repository (with datasets) is > 6 GB (download time depends on your internet speed)
* Expected run time: \~5–10 minutes on a standard computer

---

## 4. Instructions for Use

* To run individual figure scripts or customize analyses:

  * Each figure has its own script (e.g., `fig3_model_comparison.m` for Figure 3).
  * Prepare the data first by running `process_onlinedata.m`.
  * You can run any script individually after data preparation.

* Reproducing results in the manuscript:

  * Running `main.m` will reproduce all results and figures as reported in the paper.

* The scripts are modular; you can adapt or extend the analyses as needed.

---

## 5. License

This code is distributed under the MIT License.

---

## 6. Open Source Repository

* GitHub: [https://github.com/takuma929/gloss\_tinynetworks](https://github.com/takuma929/gloss_tinynetworks)
* DOI: Data will be uploaded to Zenodo or a similar repository upon publication.

---

## 7. Software Description and Documentation

* **Key operations:**
  The software processes behavioral and computational model response data, applies exclusion criteria, computes summary statistics, and generates all figures in the manuscript.

* **Fundamental tasks:**
  Raw data processing, model evaluation, analysis, and figure generation for the manuscript.

* **Algorithms and approach:**
  Standard psychophysical data processing, correlation analysis, kernel fitting, t-SNE visualization, and comparison of model predictions with human perception.

* **Dependencies:**
  Only standard MATLAB toolboxes required. Other dependencies are included in the `function` directory.
