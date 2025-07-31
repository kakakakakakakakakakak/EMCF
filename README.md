# **EFformer**
Exogenous Variable Fusional Transformer for Intersection-Level Turning Movement Traffic Flow Prediction
# Key Designs
:bulb: **Variable Fusion Layer** :Differentially integrating exogenous variables with exogenous variables.
![image](https://github.com/kakakakakakakakakakak/EMCF/blob/main/FIG/VFL.pdf)


![image](https://github.com/kakakakakakakakakakak/EMCF/blob/main/FIG/emcf.pdf)


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

The experiments in this paper use several SOTA models for comparison, and the model code comes from the Time-Series-Library-main project in GITHUB.  We thank the project for its contribution to the further development of time series and its example of a more open research environment!
We will gradually open part of the code and part of the dataset. The source code will be opened after the paper is accepted. It is currently being organized.
