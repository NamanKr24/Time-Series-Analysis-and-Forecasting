# Time Series Analysis and Forecasting

This project explores advanced techniques for **time series forecasting** by leveraging a combination of **statistical models**, **deep learning architectures**, **transformer-based models**, and **AutoML approaches**. It further culminates in the creation of a **Stacked Ensemble Hybrid Model** combining these individual methods for superior forecasting accuracy.

## Project Overview

The project focuses on implementing, analyzing, and combining the following models:
- **ARIMA** (AutoRegressive Integrated Moving Average)
- **LSTM** (Long Short-Term Memory Networks)
- **Transformer Models** (attention-based architectures)
- **Liquid ML** (AutoML framework)

Each model has been trained and evaluated on four diverse datasets:
- Synthetic Univariate Time Series
- Synthetic Multivariate Time Series
- Tesla Stock Prices (Open and Close values)
- Weather Data (Air Quality Measurements)

Finally, a **Stacked Ensemble Model** was built using the outputs of these four models to produce the final predictions.

---

## Files and Structure

- **Sales_Analysis_Forecasting.ipynb**  
  → ARIMA model implementation and forecasting.

- **RNN.ipynb**  
  → RNN and LSTM model development and evaluation.

- **Transformer.ipynb**  
  → Transformer-based model for sequential forecasting tasks.

- **LiquidML.ipynb**  
  → Liquid ML-based AutoML solution applied to time series data.

- **Final_Hybrid_Model.ipynb** 
  → Stacked Ensemble Model combining ARIMA, LSTM, Transformer, and Liquid ML outputs.

---

## Techniques Implemented

- **Statistical Modeling:**  
  Applying ARIMA to model linear trends and seasonality in time series data.

- **Deep Learning:**  
  Building Simple RNN and LSTM models to capture complex temporal patterns.

- **Attention Mechanisms:**  
  Implementing Transformer-based models to overcome the limitations of traditional RNNs by focusing on attention weights.

- **AutoML (Liquid ML):**  
  Leveraging automation to generate optimized pipelines without manual hyperparameter tuning.

- **Stacked Ensemble Hybrid Model:**  
  A final hybrid model was created by combining predictions from ARIMA, LSTM, Transformer, and Liquid ML models.  
  A **Random Forest Regressor** was used as a meta-learner, trained on the outputs of the base models to produce more robust and accurate forecasts.

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/NamanKr24/Time-Series-Analysis-and-Forecasting.git
   ```
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
   *(You may need to create a `requirements.txt` file listing libraries like `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `tensorflow`, `torch`, etc.)*

3. Open and run the Jupyter notebooks sequentially.

---

## Results Overview

- **Individual Models:**  
  Each model (ARIMA, LSTM, Transformer, Liquid ML) produced competitive results when evaluated separately on the datasets.

- **Stacked Ensemble Model:**  
  The final stacked ensemble significantly improved prediction accuracy (evaluated using RMSE), demonstrating the strength of hybridizing different approaches for time series forecasting.

---

## Future Work

- Test the stacked model on more complex and longer time series datasets.
- Explore other meta-learners like XGBoost or LightGBM for ensemble learning.
- Automate the full pipeline using a pipeline orchestration tool like MLflow.

---

## Author

- **Naman Kumar**  
  Bachelor of Technology (Computer Science and Engineering)

---

## License

This project is intended for **academic and educational** use only.
