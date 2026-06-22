# Machine Learning: Student Performance Early Warning System

A production-ready predictive analytics pipeline built in Python to identify at-risk students based on behavioral engagement metrics. This project utilizes a Random Forest ensemble model to transition from reactive academic tracking (looking at past grades) to proactive intervention (predicting outcomes based on current study habits).

## 🚀 Key Insights & Feature Importance
The model's native Gini impurity scoring revealed counter-intuitive insights for educational policy:
* **Study Hours Dominance (97% Impact):** Weekly self-study time overwhelmingly drives academic outcomes, overriding physical class attendance.
* **Attendance Paradox (2% Impact):** While foundational, physical attendance alone is a weak predictor of success if not paired with independent study.
* **Participation Irrelevance (1% Impact):** Classroom engagement tiers (Low/Medium/High) had a statistically negligible impact on final risk classification.

## 🧠 System Architecture

The codebase is strictly structured into a four-stage machine learning pipeline:

1. **Data Acquisition Module (DAM):** Ingests raw CSV data and performs stratified sampling (50,000 records) to optimize training efficiency without losing statistical significance. Converts raw grades into three actionable intervention tiers: `High Distinction`, `Pass/Medium`, and `At-Risk`.
2. **Preprocessing & Feature Engineering Engine (PFEE):** Aggressively prevents data leakage by dropping lagging indicators (`total_score`). Transforms categorical participation data via Label Encoding and normalizes continuous behavioral metrics (Attendance, Study Hours) using Z-score standardization (`StandardScaler`).
3. **Predictive Modeling Engine (PME):** Evaluates the data using an 80/20 train-test split. The core algorithm is a `RandomForestClassifier` (100 estimators, max depth of 7) chosen for its robustness against class imbalance and resistance to overfitting. 
4. **Performance Analytics Module (PAM):** Outputs eXplainable AI (XAI) visualizations, including Confusion Matrices, Prediction vs. Actual distribution bars, and Feature Importance rankings using `matplotlib` and `seaborn`.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Random Forest, Preprocessing, Metrics)
* **Data Visualization:** `matplotlib`, `seaborn`

## ⚙️ How to Run Locally

1. Clone this repository to your local machine.
2. Ensure you have the required Python libraries installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
