# Prompt Tuning the Unified Time Series Model (UniTS) model



[**Original UniTS GitHub Repo**](https://github.com/mims-harvard/UniTS) | [**Project Page**](https://zitniklab.hms.harvard.edu/projects/UniTS/)  |   [**Paper link**](https://arxiv.org/pdf/2403.00131.pdf) **(Neurips 2024)**

## Overview



<p align="center">
    <img src="https://zitniklab.hms.harvard.edu/img/UniTS-1.png" alt="UniTS-1" width="500">
</p>

### **Team Members:**
Lucy Chen, Maggie Bao, Xingli Yu

### **Project Description**
The unified multi-task time series model, or UniTS, is a state-of-the-art deep learning model designed for unified multi-task time series analysis, including classification, forecasting, imputation, and anomaly detection. These tasks have significant applications in biomedicine, especially with commonly analyzed health data like the Electrocardiogram (ECG). A biomedicine-specific UniTS model could improve clinical diagnostic robustness by accurately classifying ECG signals to detect early arrhythmias, forecast adverse cardiac events, and impute for missing data points in a diagnostic read. Therefore, this project aims to adapt UniTS for ECG-specific physiological time-series analysis by prompt-tuning it on open-source ECG datasets from PhysioNet. By applying labeled ECG data from 4 PhysioNet datasets, including the Apnea-ECG database, MIT-BIH malignant ventricular ectopy database, and the PTB-XL database sampled at 100 and 500 Hz, we assessed the performance of prompt-tuned UniTS on three different tasks against four top-performing benchmarking machine learning models for time series data. Using standard metrics like the accuracy score and mean standard error, we discovered that prompt-tuned UniTS demonstrated good performance on the forecasting task and did relatively well on the imputation task, but its performance on the classification task was outperformed by most of the benchmarking models. Based on this project, our results demonstrated the real-world implications in adapting the UniTS model to physiological time-series ECG data.

## Setups

### 1. Requirements
 Install Pytorch2.0+ and the required packages.
 Python 3.11.12 and L4 GPU were used on Google Colaboratory for this project
```
pip install -r requirements.txt
```

### 2. Prepare data
Download the Apnea data from PhysioNet (https://physionet.org/content/apnea-ecg/1.0.0/). Takes around ~5min
```
bash download_data.sh
```

Preprocess the raw ECG data into .ts file for classification and .csv files for forecasting and imputation (~5-10min)
```
python preprocessing.py
```

### 3. Download pretrained checkpoint from UniTS release 
Release link: https://github.com/mims-harvard/UniTS/releases/tag/ckpt
```
mkdir checkpoints
mv units_x128_pretrain_checkpoint.pth ./checkpoints/units_x128_pretrain_checkpoint.pth
```

### 4. Obtain wandb API key (optional)
Link: https://wandb.ai/site

### 5. Prompt-tune model (also included in the main.ipynb file)

- Few-shot transfer learning on new classification task
```
bash ./scripts/few_shot_newdata/UniTS_prompt_tuning_few_shot_newdata_classification_pct20.sh
```

- Few-shot transfer learning on new forecasting task
```
bash ./scripts/few_shot_newdata/UniTS_prompt_tuning_few_shot_newdata_forecasting_pct20.sh
```

- Few-shot transfer learning on new imputation task
```
bash ./scripts/few_shot_imputation/UniTS_prompt_tuning_few_shot_imputation_mask025.sh
```

### 6. Compare with other time series models 
iTransformer, PatchTST, TimesNet, and DLinear models were trained and evaluated using [Time-Series-Library](https://github.com/thuml/Time-Series-Library). 


## Acknowledgement
This code is built based on the [UniTS](https://github.com/mims-harvard/UniTS). Thanks!
