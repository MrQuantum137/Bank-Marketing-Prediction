# Bank Marketing Campaign Prediction using XGBoost

This project implements a high-performance machine learning pipeline to predict whether a client will subscribe to a bank term deposit based on marketing campaign data.

## 🚀 Technical Highlights
* **Advanced Pipeline:** Integrated `TargetEncoder` and `XGBClassifier` into a single Scikit-Learn Pipeline to prevent data leakage.
* **Hyperparameter Tuning:** Utilized **Bayesian Optimization** (`BayesSearchCV`) instead of standard GridSearch for more efficient and intelligent parameter discovery.
* **Categorical Handling:** Applied target encoding to high-cardinality features like `job_type` and `education` to maximize the model's predictive power.
* **Performance Metric:** Optimized for **ROC-AUC** to ensure a balanced evaluation of the model's ability to distinguish between classes.

## 🛠️ Tech Stack
* **Language:** Python
* **Key Libraries:** `XGBoost`, `Scikit-learn`, `Scikit-optimize` (skopt), `Pandas`, `Matplotlib`.

## 📊 Results
The model achieved a **Test ROC-AUC score of ~0.785**, demonstrating strong predictive capabilities. Feature importance analysis revealed that social and economic indicators were primary drivers of the prediction.

## 📂 Project Structure
* `test prototype.ipynb`: The complete development workflow from EDA to optimization.
* `bank-additional-full.csv`: The primary dataset (Sourced from UCI Machine Learning Repository).
