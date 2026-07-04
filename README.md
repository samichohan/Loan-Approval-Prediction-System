# 💰 Loan Approval Prediction System

A machine learning-based web application that predicts whether a loan application will be **approved** or **rejected**, based on applicant demographics, financial profile, and credit history. Built with Scikit-learn and deployed as an interactive web app using Streamlit.

🔗 **Live Demo:** [Add your Streamlit app link here]

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

---

## 🔍 Overview

Loan approval decisions traditionally rely on manual underwriting, which can be slow and inconsistent. This project builds a supervised machine learning pipeline that automates the initial risk assessment by analyzing applicant data — including income, employment history, credit score, and loan details — to predict the likelihood of loan approval.

The final model is served through a clean, interactive Streamlit interface, allowing users to input applicant details and receive an instant prediction along with a confidence score.

---

## ✨ Features

- Interactive web UI for real-time loan approval prediction
- Model confidence score displayed alongside the prediction
- End-to-end ML pipeline (preprocessing + classification) built with Scikit-learn
- Dataset overview and summary statistics within the app
- Trained and evaluated across multiple classification algorithms

---

## 🛠 Tech Stack

| Category            | Tools/Libraries                          |
|---------------------|-------------------------------------------|
| Language            | Python                                    |
| Data Processing     | Pandas, NumPy                             |
| Machine Learning    | Scikit-learn                              |
| Model Serialization | Joblib                                    |
| Web Framework       | Streamlit                                 |
| Development         | Jupyter Notebook / Google Colab           |

---

## 📂 Project Structure

```
Loan-Approval-Prediction-System/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── .gitignore
├── README.md
│
├── data/
│   ├── loan_data.csv                # Raw dataset
│   └── clean_loan.csv               # Cleaned dataset used for the UI
│
├── model/
│   └── loan_pipeline.pkl            # Trained preprocessing + classification pipeline
│
└── notebook/
    └── loan_model_training.ipynb    # Full EDA, preprocessing & model training notebook
```

---

## 📊 Dataset

The dataset consists of **45,000 loan applications** with the following features:

| Feature                           | Description                                      |
|------------------------------------|---------------------------------------------------|
| `person_age`                      | Applicant's age                                   |
| `person_gender`                   | Applicant's gender                                |
| `person_education`                | Highest education level                           |
| `person_income`                   | Annual income                                     |
| `person_emp_exp`                  | Years of employment experience                    |
| `person_home_ownership`           | Home ownership status (RENT, OWN, MORTGAGE, etc.) |
| `loan_amnt`                       | Requested loan amount                             |
| `loan_intent`                     | Purpose of the loan                               |
| `loan_int_rate`                   | Interest rate on the loan                         |
| `loan_percent_income`             | Loan amount as a percentage of income             |
| `cb_person_cred_hist_length`      | Length of credit history (years)                  |
| `credit_score`                    | Applicant's credit score                          |
| `previous_loan_defaults_on_file`  | Whether the applicant has previously defaulted    |
| `loan_status` (target)            | 1 = Approved, 0 = Rejected                        |

---

## ⚙️ Methodology

1. **Data Cleaning & Validation** — Checked for missing values and duplicates (dataset was already clean).
2. **Feature Encoding**
   - `person_education` was ordinally encoded (High School → Doctorate).
   - Remaining categorical features (`person_gender`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`) were one-hot encoded.
   - Numerical features were standardized using `StandardScaler`.
3. **Pipeline Construction** — Built a unified `ColumnTransformer` + classifier pipeline using Scikit-learn's `Pipeline` API, ensuring consistent preprocessing between training and inference.
4. **Model Comparison** — Trained and evaluated four classification algorithms:
   - Logistic Regression
   - Random Forest
   - Bagging Classifier
   - Extra Trees Classifier
5. **Model Selection** — Selected the best-performing model based on test accuracy.
6. **Deployment** — Serialized the final pipeline with Joblib and deployed via Streamlit.

---

## 📈 Model Performance

| Model                 | Accuracy | Precision (Approved) | Recall (Approved) | F1-Score (Approved) |
|------------------------|----------|------------------------|----------------------|------------------------|
| Logistic Regression    | 89.9%    | 0.79                   | 0.75                 | 0.77                   |
| Random Forest          | 92.5%    | 0.89                   | 0.76                 | 0.82                   |
| **Bagging Classifier** | **93.3%**| **0.89**               | **0.80**             | **0.84**               |
| Extra Trees            | 92.1%    | 0.86                   | 0.77                 | 0.81                   |

✅ **Bagging Classifier** was selected as the final model based on its highest overall accuracy and balanced precision-recall performance on the minority (approved) class.

---

## 🚀 Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/samichohan/Loan-Approval-Prediction-System.git
cd Loan-Approval-Prediction-System

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`) in your browser.

---

## 🔮 Future Improvements

- Add SHAP/feature importance visualizations for model explainability
- Hyperparameter tuning via GridSearchCV/Optuna
- Handle class imbalance with SMOTE or class weighting
- Add model versioning and experiment tracking (MLflow)
- Deploy via Docker for platform-independent reproducibility

---

## 👤 Author

**Sami Chohan**
GitHub: [@samichohan](https://github.com/samichohan)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).