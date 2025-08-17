# Loan_approval_system
A machine learning model to build an equitable credit scoring engine for the "credit invisible" gig economy, with a focus on solving class imbalance to ensure fair and effective risk prediction.
# Project Nova: An Equitable Credit Scoring Engine

This project is a proof-of-concept for an equitable credit scoring model designed for gig economy workers who are often "credit invisible" due to a lack of formal credit histories. The model predicts creditworthiness based on financial behavior and performance metrics, aiming to provide a fairer pathway to financial products.

This solution was developed based on the "Project Nova" problem statement, using the Lending Club dataset as a robust, large-scale proxy.

## The Core Challenge: Class Imbalance

The primary technical challenge was predicting loan defaults, which are rare events in the dataset. A standard model would achieve high accuracy by simply predicting that no one will default, making it useless for risk assessment. Our main focus was to build a model that could effectively identify this small but critical group of high-risk applicants.

## Our Methodology

Our end-to-end machine learning pipeline consists of several key stages:

1.  **Data Preparation**: We started with a large dataset of over 1 million loans and took a 10% random sample for efficient development. The data was rigorously cleaned, and we filtered for loans with definitive outcomes ('Fully Paid' or 'Charged Off') to create a clear target for our model.

2.  **Feature Engineering**: We created new, more powerful features to provide the model with deeper insights. For example, the `loan_to_income_ratio` was engineered to give the model crucial context about the loan's size relative to the applicant's income, which proved to be a top predictor.

3.  **Solving Class Imbalance**: This was the most critical step. We used the **`scale_pos_weight`** parameter in our LightGBM model, calculated as the ratio of non-defaults to defaults. This technique forces the model to pay significantly more attention to the rare default cases, transforming it from a passive observer into an active risk detector.

## Final Results

The implemented strategies were highly effective. Our final model shows a dramatic improvement over a standard approach:
* **Default Detection (Recall)**: We increased the model's ability to identify actual defaults **from 4% to nearly 60%**.
* **Trade-off**: This was achieved with an accepted trade-off in precision and overall accuracy, resulting in a balanced model that is genuinely useful for its business case.

## Tech Stack

* **Language**: Python 3.x
* **Core Libraries**: Pandas, NumPy, Scikit-learn
* **Modeling**: LightGBM
* **Visualization**: Matplotlib, Seaborn
* **Hyperparameter Tuning**: Optuna

## How to Run the Project

### 1. Prerequisites
Ensure you have Python 3 installed.

### 2. Installation
Clone the repository and install the required libraries using the following command:
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn optuna
