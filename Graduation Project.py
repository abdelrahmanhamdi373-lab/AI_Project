import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. Load Dataset
# =========================================================

df = pd.read_csv("aug_train.csv")

# =========================================================
# 1. Explore Data
# =========================================================
'''
df.head()
df.info()
df.columns
df.shape
df.describe()
df.isnull().sum()
df.duplicated().sum()
df['gender'].unique()
'''

# =========================================================
# 2. Data Cleaning
# =========================================================

df['gender'] = df['gender'].str.lower()
df['gender'] = df['gender'].replace('other', np.nan)
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

df['enrolled_university'] = df['enrolled_university'].fillna(
    df['enrolled_university'].mode()[0]
)

df['education_level'] = df['education_level'].fillna(
    df['education_level'].mode()[0]
)

df['major_discipline'] = df['major_discipline'].replace(
    ['No Major', 'Other'],
    np.nan
)

df['major_discipline'] = df['major_discipline'].fillna(
    df['major_discipline'].mode()[0]
)

df['experience'] = df['experience'].fillna(
    df['experience'].mode()[0]
)

df['company_size'] = df['company_size'].fillna(
    df['company_size'].mode()[0]
)

df['company_type'] = df['company_type'].replace(
    'Other',
    np.nan
)

df['company_type'] = df['company_type'].fillna(
    df['company_type'].mode()[0]
)

df['last_new_job'] = df['last_new_job'].fillna(
    df['last_new_job'].mode()[0]
)

# =========================================================
# 3.Data Analyzing
# =========================================================

# Gender
df.groupby('gender')['target'].mean() * 100


# City
df.groupby('city')['target'].mean() * 100


# City Development Index
df.groupby('target')['city_development_index'].mean()


# Relevant Experience
df.groupby('relevent_experience')['target'].mean() * 100


# Enrolled University
df.groupby('enrolled_university')['target'].mean() * 100


# Education Level
df.groupby('education_level')['target'].mean() * 100


# Major Discipline
df.groupby('major_discipline')['target'].mean() * 100


# Experience
df.groupby('experience')['target'].mean() * 100


# Company Size
df.groupby('company_size')['target'].mean() * 100


# Company Type
df.groupby('company_type')['target'].mean() * 100


# Last New Job
df.groupby('last_new_job')['target'].mean() * 100


# Training Hours
df.groupby('training_hours')['target'].mean()


# Save cleaned data
df.to_csv('cleaned_data.csv', index=False)


# =========================================================
# 3. Define Categorical Columns
# =========================================================

categorical_cols = [
    'city',
    'gender',
    'relevent_experience',
    'enrolled_university',
    'education_level',
    'major_discipline',
    'experience',
    'company_size',
    'company_type',
    'last_new_job'
]


# =========================================================
# 4. Features and Target
# =========================================================

X = df.drop('target', axis=1)
y = df['target']

# Remove ID
X = X.drop('enrollee_id', axis=1)


# =========================================================
# 5. Encoding
# =========================================================

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)


# =========================================================
# 6. Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# 7. Random Oversampling
#    Training Data Only
# =========================================================

ros = RandomOverSampler(random_state=42)

X_train_resampled, y_train_resampled = ros.fit_resample(
    X_train,
    y_train
)

print("Before Oversampling:")
print(y_train.value_counts())

print("\nAfter Oversampling:")
print(y_train_resampled.value_counts())


# =========================================================
# 8. Scaling
# =========================================================

scaler = StandardScaler()

X_train_resampled[
    ['city_development_index', 'training_hours']
] = scaler.fit_transform(
    X_train_resampled[
        ['city_development_index', 'training_hours']
    ]
)

X_test[
    ['city_development_index', 'training_hours']
] = scaler.transform(
    X_test[
        ['city_development_index', 'training_hours']
    ]
)


# =========================================================
# 9. Logistic Regression
# =========================================================

model_lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model_lr.fit(
    X_train_resampled,
    y_train_resampled
)

y_pred_lr = model_lr.predict(X_test)


# =========================================================
# 10. Logistic Regression Evaluation
# =========================================================

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

cm_lr = confusion_matrix(y_test, y_pred_lr)

print("\n" + "=" * 50)
print("Logistic Regression Result")
print("=" * 50)

print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1 Score:", f1_lr)

print("\nConfusion Matrix:")
print(cm_lr)


# =========================================================
# 11. Random Forest + Grid Search
# =========================================================

model_rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 15, 20],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(
    estimator=model_rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_rf.fit(
    X_train_resampled,
    y_train_resampled
)


# =========================================================
# 12. Best Random Forest Model
# =========================================================

model_rf = grid_rf.best_estimator_

print("\n" + "=" * 50)
print("Random Forest - Grid Search")
print("=" * 50)

print("Best Parameters:")
print(grid_rf.best_params_)

print("Best Cross-Validation F1:")
print(grid_rf.best_score_)


# =========================================================
# 13. Random Forest Prediction
# =========================================================

y_pred_rf = model_rf.predict(X_test)


# =========================================================
# 14. Random Forest Evaluation
# =========================================================

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\n" + "=" * 50)
print("Random Forest Final Result")
print("=" * 50)

print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)

print("\nConfusion Matrix:")
print(cm_rf)


# =========================================================
# 15. Model Comparison
# =========================================================

print("\n" + "=" * 50)
print("Model Comparison")
print("=" * 50)

print("Logistic Regression F1:", f1_lr)
print("Random Forest F1:", f1_rf)

if f1_rf > f1_lr:
    print("\nBest Model: Random Forest")
else:
    print("\nBest Model: Logistic Regression")


# =========================================================
# 16. Random Forest Confusion Matrix
# =========================================================

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()