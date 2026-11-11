# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import(StandardScaler, PolynomialFeatures, OneHotEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.svm import SVC, SVR
from sklearn.metrics import(mean_absolute_error,mean_squared_error,
    r2_score,accuracy_score,precision_score,recall_score,
    f1_score,confusion_matrix,classification_report,log_loss,roc_auc_score)


import warnings
warnings.filterwarnings("ignore")

# STEP 2: LOAD DATASET
df = pd.read_csv("Sub_Division_IMD_2017.csv")
print("Dataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
# STEP 3: DATA PREPROCESSING

df["JAN_JUN_TOTAL"] = df[["JAN","FEB","MAR","APR","MAY","JUN"]].sum(axis=1,min_count=6)
df = df.dropna(subset=["ANNUAL", "JAN_JUN_TOTAL"])

# Regression features
features=["SUBDIVISION","YEAR","JAN","FEB","MAR","APR","MAY","JUN"]
X = df[features]
y = df["ANNUAL"]

# Chronological train-test split
# Train on earlier years, test on later years

cutoff = df["YEAR"].quantile(0.80)

train_mask = df["YEAR"] <= cutoff
test_mask = df["YEAR"] > cutoff

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Training through year:", cutoff)

# STEP 4: PREPROCESSING PIPELINES

numeric_features = [
    "YEAR", "JAN", "FEB", "MAR", "APR", "MAY", "JUN"
]

categorical_features = ["SUBDIVISION"]


def create_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

# REGRESSION MODELS
regression_models = {
# MODEL 1: SIMPLE LINEAR REGRESSION
    "Simple Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())]),

# MODEL 2: MULTIPLE LINEAR REGRESSION
    "Multiple Linear Regression": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", LinearRegression())]),

# MODEL 3: POLYNOMIAL REGRESSION
    "Polynomial Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())]),

# MODEL 4: DECISION TREE REGRESSOR
    "Decision Tree Regressor": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", DecisionTreeRegressor(
            max_depth=6, random_state=42))]),

# MODEL 5: SUPPORT VECTOR REGRESSION
    "Support Vector Regression": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", SVR(kernel="rbf", C=10))])}

# Simple and Polynomial Regression use one feature
X_simple_train = X_train[["JAN"]]
X_simple_test = X_test[["JAN"]]

# A more informative single feature for polynomial regression
X_poly_train = df.loc[train_mask, ["JAN_JUN_TOTAL"]]
X_poly_test = df.loc[test_mask, ["JAN_JUN_TOTAL"]]
regression_predictions = {}
regression_results = []
for name, model in regression_models.items():
    if name == "Simple Linear Regression":
        xtr, xte = X_simple_train, X_simple_test
    elif name == "Polynomial Regression":
        xtr, xte = X_poly_train, X_poly_test
    else:
        xtr, xte = X_train, X_test
    model.fit(xtr, y_train)
    predictions = model.predict(xte)
    regression_predictions[name] = predictions
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    regression_results.append({"Model": name,"MAE": mae,"MSE": mse,"RMSE": rmse,"R2 Score": r2})
    print("\n", name)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# CLASSIFICATION TARGET CREATION
# Create rainfall categories using training data only

q1, q2 = y_train.quantile([1/3, 2/3])
bins = [-np.inf, q1, q2, np.inf]
labels = ["Low", "Medium", "High"]
y_class_train = pd.cut(
    y_train, bins=bins, labels=labels
).astype(str)

y_class_test = pd.cut(
    y_test, bins=bins, labels=labels
).astype(str)

print("\nRainfall Category Thresholds:")
print("Low:", "Below", q1)
print("Medium:", q1, "to", q2)
print("High:", "Above", q2)

# CLASSIFICATION MODELS

classification_models = {

# MODEL 6: LOGISTIC REGRESSION
    "Logistic Regression": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

# MODEL 7: DECISION TREE CLASSIFIER
    "Decision Tree Classifier": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", DecisionTreeClassifier(
            max_depth=6, random_state=42
        ))
    ]),

# MODEL 8: LINEAR SVM
    "Linear SVM": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", SVC(
            kernel="linear", probability=True, random_state=42
        ))
    ]),

# MODEL 9: RBF SVM
    "RBF SVM": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", SVC(
            kernel="rbf", probability=True, random_state=42
        ))
    ]),

# MODEL 10: POLYNOMIAL SVM
    "Polynomial SVM": Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", SVC(
            kernel="poly", degree=3,
            probability=True, random_state=42))])}

classification_predictions = {}
classification_results = {}
confusion_matrices={}
for name, model in classification_models.items():
    model.fit(X_train, y_class_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    classification_predictions[name] = predictions
    accuracy = accuracy_score(y_class_test, predictions)
    precision = precision_score(
        y_class_test, predictions,
        average="weighted", zero_division=0
    )
    
    recall=recall_score(y_class_test, predictions,average="weighted", zero_division=0)
    f1 = f1_score(y_class_test, predictions,average="weighted", zero_division=0)
    ll=log_loss(y_class_test, probabilities,labels=labels)

    try:
        auc = roc_auc_score(y_class_test,probabilities,multi_class="ovr",average="weighted",labels=labels)
    except ValueError:auc = np.nan
    cm = confusion_matrix(y_class_test, predictions,labels=labels)
    confusion_matrices[name] = cm

    classification_results[name] = {"Accuracy": accuracy,"Precision": precision,"Recall": recall,
        "F1 Score": f1,"Log Loss": ll,"AUC": auc}
    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Log Loss:", ll)
    print("AUC:", auc)
    print("\nClassification Report:")
    print(classification_report(y_class_test,predictions,labels=labels,zero_division=0))

# STEP 5: DISPLAY MODEL COMPARISON TABLES

regression_df = pd.DataFrame(regression_results)

classification_df = pd.DataFrame(
    classification_results
).T

print("\n========== REGRESSION MODEL COMPARISON ==========")
print(regression_df.round(4).to_string(index=False))

print("\n========== CLASSIFICATION MODEL COMPARISON ==========")
print(classification_df.round(4))


# STEP 6: VISUALIZATION
# GRAPH 1: CORRELATION HEATMAP
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Rainfall Correlation Heatmap")
plt.tight_layout()
plt.show()

# GRAPH 2: ANNUAL RAINFALL TREND
annual_trend = df.groupby("YEAR")["ANNUAL"].mean()
plt.figure(figsize=(12, 5))
plt.plot(annual_trend.index, annual_trend.values)
plt.title("Average Annual Rainfall Trend")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.grid()
plt.show()


# GRAPH 3: ANNUAL RAINFALL HISTOGRAM
plt.figure(figsize=(8, 5))
sns.histplot(df["ANNUAL"], kde=True)
plt.title("Annual Rainfall Distribution")
plt.xlabel("Annual Rainfall")
plt.show()


# GRAPH 4: MONTHLY RAINFALL BOXPLOT
monthly_columns = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN","JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
plt.figure(figsize=(13, 6))
sns.boxplot(data=df[monthly_columns])
plt.title("Monthly Rainfall Distribution")
plt.ylabel("Rainfall (mm)")
plt.show()


#GRAPH 5: REGRESSION MODEL COMPARISON
plt.figure(figsize=(11, 5))
sns.barplot(data=regression_df,x="Model",y="R2 Score")
plt.title("Regression Models - R2 Score Comparison")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()


#GRAPH 6: ACTUAL VS PREDICTED VALUES
fig, axes = plt.subplots(3, 2, figsize=(13, 13))
for ax, (name, predictions) in zip(
    axes.flat, regression_predictions.items()):
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.set_title(name)
    ax.set_xlabel("Actual Rainfall")
    ax.set_ylabel("Predicted Rainfall")
    ax.grid()
axes.flat[-1].axis("off")
plt.tight_layout()
plt.show()


# GRAPH 7: RESIDUAL PLOTS
fig, axes = plt.subplots(3, 2, figsize=(13, 13))
for ax, (name, predictions) in zip(
    axes.flat, regression_predictions.items()):
    residuals = y_test.values - predictions
    ax.scatter(predictions, residuals, alpha=0.5)
    ax.axhline(0, linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("Predicted Rainfall")
    ax.set_ylabel("Residuals")
axes.flat[-1].axis("off")
plt.tight_layout()
plt.show()


# GRAPH 8: CONFUSION MATRICES
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, (name, cm) in zip(
    axes.flat, confusion_matrices.items()):
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels,ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
axes.flat[-1].axis("off")
plt.tight_layout()
plt.show()


# GRAPH 9: CLASSIFICATION MODEL ACCURACY
class_plot = classification_df.reset_index()
class_plot = class_plot.rename(columns={"index": "Model"})
plt.figure(figsize=(11, 5))
sns.barplot(data=class_plot,x="Model",y="Accuracy")
plt.title("Classification Model Accuracy Comparison")
plt.xticks(rotation=35, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# GRAPH 10: DECISION TREE VISUALIZATION
tree_model = classification_models[
    "Decision Tree Classifier"]
tree = tree_model.named_steps["model"]
feature_names = (
    tree_model.named_steps["preprocessor"]
    .get_feature_names_out())
plt.figure(figsize=(22, 12))
plot_tree(tree,feature_names=feature_names,class_names=tree.classes_,filled=True,
          rounded=True,max_depth=3,fontsize=8)
plt.title("Decision Tree Classifier")
plt.show()

# STEP 7: FINAL MODEL RESULTS

print("\n'''''''''''''''''''''''''''''''''''")
print("ALL 10 MODELS EXECUTED SUCCESSFULLY")
print("'''''''''''''''''''''''''''''''''''''")
print("\nRegression Models:", len(regression_models))
print("Classification Models:", len(classification_models))
print("Total Models:",
      len(regression_models) + len(classification_models))
print("\nRegression Results:")
print(regression_df)
print("\nClassification Results:")
print(classification_df)
