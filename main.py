import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. DATA ACQUISITION MODULE (DAM)
# ==========================================
print("--- Initializing Data Acquisition ---")
# Load the dataset
df = pd.read_csv('student_performance.csv')


df = df.sample(n=50000, random_state=42).copy()


grade_mapping = {
    'A': 'High Distinction', 'B': 'High Distinction',
    'C': 'Pass/Medium',
    'D': 'At-Risk', 'F': 'At-Risk'
}
df['Risk_Level'] = df['grade'].map(grade_mapping)

# GRAPH 1: The Setup Plot (Data Distribution)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Risk_Level', order=['High Distinction', 'Pass/Medium', 'At-Risk'], palette='viridis')
plt.title('Initial Setup: Distribution of Student Risk Levels')
plt.ylabel('Number of Students')
plt.xlabel('Academic Standing')
plt.show()

# ==========================================
# 2. PREPROCESSING ENGINE (PFEE)
# ==========================================
print("--- Running Preprocessing Pipeline ---")

df_clean = df.drop(['student_id', 'total_score', 'grade'], axis=1)


df_clean['Participation_Category'] = pd.cut(
    df_clean['class_participation'], 
    bins=[-0.1, 3.5, 7.5, 10.0], 
    labels=['Low', 'Medium', 'High']
)

le = LabelEncoder()
df_clean['Participation_Encoded'] = le.fit_transform(df_clean['Participation_Category'])

X = df_clean[['attendance_percentage', 'weekly_self_study_hours', 'Participation_Encoded']]
y = df_clean['Risk_Level']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==========================================
# 3. PREDICTIVE MODELING ENGINE (PME)
# ==========================================
print("--- Training Random Forest Ensemble ---")
# 80/20 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf_model.fit(X_train, y_train)

# Generate Predictions for the test set
predictions = rf_model.predict(X_test)


# ==========================================
# 4. PERFORMANCE ANALYTICS MODULE (PAM)
# ==========================================
print("--- Generating Evaluation Metrics ---")
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# GRAPH 2: The Results Plot (Confusion Matrix)
labels = ['High Distinction', 'Pass/Medium', 'At-Risk']
cm = confusion_matrix(y_test, predictions, labels=labels)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Results Plot: Confusion Matrix (Prediction Accuracy)')
plt.xlabel('Predicted Class (AI Output)')
plt.ylabel('Actual Class (True Data)')
plt.show()

# GRAPH 3: Comparison Plot (Predicted vs Actual)
comparison_df = pd.DataFrame({'Actual Data': y_test, 'Model Predictions': predictions})
comparison_counts = comparison_df.apply(pd.Series.value_counts).fillna(0)

comparison_counts = comparison_counts.reindex(labels)

comparison_counts.plot(kind='bar', figsize=(9, 5), color=['#2ca02c', '#1f77b4'])
plt.title('Comparison Plot: Actual Student Outcomes vs Model Predictions')
plt.ylabel('Number of Students')
plt.xticks(rotation=0)
plt.legend(loc='upper right')
plt.show()

# GRAPH 4: Feature Importance 
importances = rf_model.feature_importances_
features = ['Attendance %', 'Study Hours', 'Participation Tier']

plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=features, palette='magma')
plt.title('Feature Importance: What drives the AI\'s predictions?')
plt.xlabel('Impact Weighting (Gini Score)')
plt.show()