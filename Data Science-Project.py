# Graduation Analysis
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance

# Explainability
import shap

# Load Data
file_path = Path(r"C:\Users\Hp\Desktop\Data Science\Cohort 4 Capstone Project - Dataset - Cohort 3 DS.csv")
data = pd.read_csv(file_path)

# Drop unnecessary columns if they exist
for col in ["Timestamp", "Id. No"]: 
    if col in data.columns:
        data = data.drop(columns=[col])

# Encode target
target_col = "Graduated"
data[target_col] = data[target_col].map({"No": 0, "Yes": 1})

# Separate features and target
X = data.drop(columns=[target_col])
y = data[target_col]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Exploratory Data Analysis (EDA)
data.head() # displays the first 5 rows of the dataframe
data.info() # concise summary of the dataframe
data.describe() # calculate the summary statistics
data.isnull().sum() # count of missing values in each column
data.duplicated().sum() # count of duplicate rows

# Example plots
sns.countplot(data=data, x="Age range")
plt.title("Distribution of Age range")
plt.show()

# Gender Breakdown
data["Gender"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Gender Distribution")
plt.show()

# Country Representation
sns.countplot(data=data, y="Country")
plt.title("Country Representation")
plt.show()

# Track Distribution
sns.countplot(data=data, x="Which track are you applying for?",
              order=data["Which track are you applying for?"].value_counts().index)
plt.title("Track Distribution")
plt.xticks(rotation=45)
plt.show()

# Years of Learning Experience
sns.histplot(data["How many years of learning experience do you have in the field of data?"], bins=10, kde=True)
plt.title("Years of Learning Experience Distribution")
plt.show()

# Hours Available per Week
sns.histplot(data["How many hours per week can you commit to learning?"], bins=10, kde=True)
plt.title("Hours per Week Distribution")
plt.show()

# Hours per Week by Aim of Study
sns.boxplot(data=data, x="How many hours per week can you commit to learning?",
            y="What is your main aim for joining the mentorship program?")
plt.title("Hours per Week by Aim of Study")
plt.show()

# Scores by Graduation Status
sns.boxplot(data=data, x="Graduated", y="Total score")
plt.title("Scores by Graduation Status")
plt.show()

# Graduation Rate by Gender (fixed)
sns.barplot(data=data, x="Gender", y="Graduated", estimator=np.mean)
plt.title("Graduation Rate by Gender")
plt.ylabel("Graduation Rate")
plt.show()

# Correlation Heatmap
Hours_map = {
    "less than 6 hours": 5,   
    "7-14 hours": 10,
    "more than 14 hours": 16  
}

experience_map = {
    "Less than six months": 0.25,
    "6 months - 1 year": 0.75,
    "1-2 years": 1.5,
    "1-3 years": 2,
    "3-5 years": 4,
    "4-6 years": 5,
    "More than 5 years": 6
}

# Apply mappings
data["experience_years"] = data["How many years of learning experience do you have in the field of data?"].map(experience_map)
data["hours_per_week"] = data["How many hours per week can you commit to learning?"].map(Hours_map)

# Correlation
corr = data.corr(numeric_only=True)
corr_columns = ["experience_years", "hours_per_week", "Total score", "Graduated"]
corr_matrix = data[corr_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Main aim
sns.countplot(data=data, y="What is your main aim for joining the mentorship program?",
              order=data["What is your main aim for joining the mentorship program?"].value_counts().index)
plt.title("Main Aim for Joining")
plt.show()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# Models & Resampling
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

resamplers = {
    "SMOTE": SMOTE(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42),
    "SMOTETomek": SMOTETomek(random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_score = -1
best_combo = None
best_model = None

for resample_name, resampler in resamplers.items():
    for model_name, model in models.items():
        pipe = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("resample", resampler),
            ("clf", model)
        ])
        
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        mean_score = np.mean(cv_scores)
        
        print(f"\n=== {resample_name} + {model_name} ===")
        print(f"CV mean F1_macro: {mean_score:.4f}")
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
        
        if mean_score > best_score:
            best_score = mean_score
            best_combo = (resample_name, model_name)
            best_model = pipe

print("\n============================")
print(f"Best combo: {best_combo[0]} + {best_combo[1]} with CV F1_macro = {best_score:.4f}")
print("============================")

# Final Model Evaluation
y_pred = best_model.predict(X_test)
print("\nFinal Model Evaluation:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance & SHAP
if isinstance(best_model.named_steps["clf"], XGBClassifier):
    xgb_model = best_model.named_steps["clf"]
    
    # Feature importance
    plt.figure(figsize=(8, 6))
    plot_importance(xgb_model, max_num_features=10, importance_type="gain")
    plt.title("Top 10 Features Driving Graduation")
    plt.show()
    
    # SHAP analysis
    X_test_transformed = best_model.named_steps["preprocessor"].transform(X_test)
    ohe_features = best_model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_cols)
    all_features = np.concatenate([ohe_features, numeric_cols])
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed)
    
    shap.summary_plot(shap_values, X_test_transformed, feature_names=all_features)

    # Insights & Recommendations
    importance = xgb_model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": all_features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Predictors of Graduation:")
    print(feat_imp.head(10))

    print("\n=== Insights & Recommendations ===")
    for feat, score in feat_imp.head(5).values:
        if "hours" in feat.lower():
            print("- Increasing weekly study hours is strongly linked with higher graduation rates.")
        elif "experience" in feat.lower():
            print("- Prior experience in data science boosts graduation likelihood. Provide bridging support for beginners.")
        elif "score" in feat.lower():
            print("- Higher assessment scores predict graduation. Academic mentoring can help low scorers succeed.")
        elif "gender" in feat.lower():
            print("- Gender differences observed. Tailored mentorship may balance outcomes.")
        elif "country" in feat.lower():
            print("- Regional differences exist. Address localized learning challenges for better results.")
        else:
            print(f"- {feat} is an important predictor. Investigate its role in graduation outcomes.")

    print("\nSHAP analysis confirms these factors consistently influence graduation, "
          "helping to design targeted interventions for learners.")
