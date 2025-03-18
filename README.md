# **EMCF**
Exogenous Variable Fusional Multi-Scale Cross-Dimension Framework for Intersection-Level Turning Movement Traffic Flow Prediction
# Key Designs
:bulb: **Variable Fusion Layer** :Differentially integrating exogenous variables with exogenous variables.
![image](https://github.com/user-attachments/assets/537b6517-4410-488f-96f6-89c432d42f0f)
:bulb: **Multi-Scale Segmentation Mechanism** :Optimizing lookback window length is essential for robust forecasting.
A multi-scale segmentation mechanism partitions input sequences into different scales, processes them independently, 
and aggregates the outputs to capture multi-scale temporal dependencies, improving prediction accuracy and model robustness.
![image](https://github.com/user-attachments/assets/9f54601e-d04f-4851-be93-f72b869d3b30)
In addition to the above, we have innovated in the encoder by incorporating a **single-channel MAMBA** (state space mechanism) and a cross-dimensional attention mechanism. An enhanced LSTM(**En-LSTM**) has also been developed.
# Dataset- Robust and Outstanding dataset

First, abundant turning movement flow data are collected across multiple regions in Ontario, Canada. An overview of the datasets and their contents is presented in Table 1.
![image](https://github.com/user-attachments/assets/a049ccc3-d59d-4af0-b0ea-67df58bd84ad)

These datasets contain long-term traffic records from various urban intersections. The data volume refers to the number of recorded entries, where M denotes million.
These rich datasets provide us with a solid experimental foundation to fully validate the model's performance.

We provide you with a convenient installation environment

```
pip install -r requirements.txt
```
In addition, we provide an installation package for mamba_ssm, which solves the installation problem

# Experimental Analysis
This paper conducted multiple experiments to verify the comprehensive performance of the model.


We will gradually open part of the code and part of the dataset. The source code will be opened after the paper is accepted. It is currently being organized.
