
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Set a random seed for reproducibility
np.random.seed(42)

# -----------------------------
# 1. Core AutoML Functions
# -----------------------------

def detect_task_type(target_series):
    """Detects if the task is classification or regression."""
    # Heuristic: if the number of unique values is low and the data type is integer or object,
    # it's likely a classification problem.
    if target_series.dtype == 'object' or target_series.nunique() < len(target_series) * 0.1:
        return 'Classification'
    else:
        return 'Regression'

def train_and_evaluate_models(X, y, task_type):
    """
    Trains and evaluates a set of models based on the task type.
    Returns a dictionary of model performances.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary to store performance results
    results = {}

    if task_type == 'Classification':
        # Define classification models to test
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {"Accuracy": accuracy}
            except Exception as e:
                results[name] = {"Error": str(e)}

    elif task_type == 'Regression':
        # Define regression models to test
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results[name] = {"R-squared": r2, "RMSE": rmse}
            except Exception as e:
                results[name] = {"Error": str(e)}

    return results

# -----------------------------
# 2. Streamlit Interface
# -----------------------------

st.title("Automated Machine Learning Model Selector")
st.markdown("Upload a CSV, select a target column, and let the app automatically train and compare different models.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Get a list of column names for the selectbox
    columns = df.columns.tolist()

    # User selects the target variable
    target_column = st.selectbox("Select the target variable (column to predict):", columns)

    if st.button("Run AutoML"):
        if target_column:
            st.write(f"Running AutoML for target variable: **{target_column}**...")

            # Separate features (X) and target (y)
            y = df[target_column]
            X = df.drop(columns=[target_column])

            # Handle non-numeric features by one-hot encoding
            X = pd.get_dummies(X, drop_first=True)

            # Handle string targets in classification
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Detect task type
            task_type = detect_task_type(y)
            st.success(f"Task detected: **{task_type}**")

            # Train and evaluate models
            with st.spinner('Training and evaluating models...'):
                results = train_and_evaluate_models(X, y, task_type)

            # Display results
            st.subheader("Model Performance Results")
            if task_type == 'Classification':
                results_df = pd.DataFrame(results).T.sort_values(by="Accuracy", ascending=False)
                st.dataframe(results_df)
                st.markdown("---")
                st.subheader("Best Model (by Accuracy)")
                best_model = results_df.index[0]
                best_score = results_df.iloc[0]["Accuracy"]
                st.success(f"The best model is **{best_model}** with an accuracy of **{best_score:.2f}**.")
            
            elif task_type == 'Regression':
                results_df = pd.DataFrame(results).T.sort_values(by="R-squared", ascending=False)
                st.dataframe(results_df)
                st.markdown("---")
                st.subheader("Best Model (by R-squared)")
                best_model = results_df.index[0]
                best_score = results_df.iloc[0]["R-squared"]
                st.success(f"The best model is **{best_model}** with an R-squared of **{best_score:.2f}**.")
        else:
            st.error("Please select a target variable to run the AutoML process.")

