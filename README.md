# chronos-project
# Retail Demand Forecasting: Zero-Shot Multivariate Inference with Chronos-T5 

### Executive Overview
Retail sales data is notoriously difficult to model due to its "sparse" nature (intermittent demand with frequent zero-sales days) and heavy reliance on external causal factors like pricing elasticity and calendar events. Traditional deterministic models (e.g., ARIMA, standard LSTMs) typically fail under these non-stationary conditions, often predicting inaccurate flat averages.

This repository contains a multivariate forecasting pipeline designed to solve the "Causal Tangle" of retail demand. By leveraging Amazon's Chronos-T5 foundation model in a zero-shot capacity, the pipeline processes erratic historical data alongside dynamic pricing and temporal covariates to generate a calibrated Probabilistic Risk Envelope for inventory safety-stock planning.

### Core Architecture and Methodology

#### 1. The Causal Tangle (Covariate Engineering)
A univariate forecast lacks the context to predict spikes driven by external factors. This pipeline engineers dynamic covariates from the Walmart M5 dataset:
* Target: Daily item sales (melted from wide to long format).
* Temporal Covariate: Binary categorical flags for cultural and national holidays.
* Dynamic Covariate: Weekly historical sell prices to map price elasticity.

#### 2. Robust Local Scaling (arcsinh)
Foundation models struggle with extreme variance. To stabilize the attention mechanism without breaking on the exact 0 values native to retail data, the pipeline applies an Inverse Hyperbolic Sine transformation. This standardizes the data and acts as a mathematical shock absorber for massive promotional spikes.

#### 3. Tensor Construction
Sequences are stacked into a 3D PyTorch Tensor configured for the Chronos Alternating Attention mechanism: [Batch, Variates, Sequence Length]. The context length is optimized to capture robust year-over-year seasonality.

#### 4. Probabilistic Inference
Rather than predicting a single deterministic median, the model generates 50 distinct future paths over a 28-day horizon. These paths are mathematically aggregated to extract a P10 (Floor), P50 (Median), and P90 (Ceiling) risk envelope.

### Performance Evaluation: Weighted Quantile Loss (wQL)
Standard metrics like Mean Absolute Error (MAE) break down on sparse data. This model is evaluated using Mean Weighted Quantile Loss (wQL), which penalizes the model based on its ability to encapsulate reality within its P10-P90 probability cone. 

**Zero-Shot Baseline Results:**
* P10 wQL (Stockout Risk): 0.6815 — Successfully recognizes sparsity without hallucinating phantom sales.
* P90 wQL (Overstock Risk): 0.8832 — Effectively expands the cone of uncertainty to catch promotional spikes.
* Mean wQL Score: 0.8378

Note: In the domain of zero-shot intermittent retail forecasting, a Mean wQL under 1.0 indicates a highly viable production baseline for supply chain operations.

### Tech Stack
* Modeling: Hugging Face transformers, amazon/chronos-t5-base, PyTorch
* Data Engineering: Pandas, NumPy
* Evaluation: Scikit-learn, Custom wQL Math
* Visualization: Plotly (Dual-axis interactive causal mapping)
