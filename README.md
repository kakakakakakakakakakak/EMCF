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
# Dataset

First, abundant turning movement flow data are collected across multiple regions in Ontario, Canada. An overview of the datasets and their contents is presented in Table 1.



We will gradually open part of the code and part of the dataset. The source code will be opened after the paper is accepted. It is currently being organized.
