import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.feature_engineer import FeatureEngineer
from src.bank_loan_cleaner import BankLoanCleaner

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        X = df.drop(columns=['loan_status'])
        y = df['loan_status']
        print(f"Data loaded successfully from {file_path}.")
        return X, y
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

def build_pipeline() -> Pipeline:
    """Build a machine learning pipeline."""
    # Define the preprocessing steps for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
            ('cat', categorical_transformer, make_column_selector(dtype_include=object))
        ],
        remainder='passthrough',  # Keep any remaining columns unchanged
        verbose_feature_names_out=False
    )

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('cleaner', BankLoanCleaner()),
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    pipeline.set_output(transform="pandas")  # Ensure the output is a pandas DataFrame
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return accuracy, precision, and recall."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Model Evaluation:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}")
    
    # Explicitly log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

def log_feature_importance(model):
    """" Calculates, prints, and returns the top 3 feature importances."""
    feature_names = model[:-1].get_feature_names_out()  # Get feature names from the preprocessor
    importances = model[-1].feature_importances_  # Get feature importances from the classifier
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("Top 3 Feature Importances:")
    print(feature_importance_df.head(3).to_string(index=False))

    # Log a JSON artifact of feature importances to MLflow
    mlflow.log_table(data=feature_importance_df, artifact_file="feature_importances_table.json")

def main():
    # --- MLflow Experiment Setup ---
    # Group all related runs under a single experiment for better organization and tracking.
    mlflow.set_experiment("Bank_Loan_Classification")

    # Enable autologging for scikit-learn (captures GridSearch parameters + CV results)
    mlflow.sklearn.autolog(
        log_models=False,  # Log the best model found during GridSearchCV
        log_input_examples=True,  # Log input examples for reproducibility
        log_model_signatures=True  # Log model signatures for input/output schema
    )
    
    #1. Load Data
    data_path = "data/loan_data.csv"
    X, y = load_data(data_path)
    
    #2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #3. Build and Tune Pipeline
    pipeline = build_pipeline()
    print("Pipeline built successfully.")

    param_grid = {
        'classifier__n_estimators': [10, 50, 100, 150],
        'classifier__max_depth': [2, 5, 10, 15],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4, 8]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    with mlflow.start_run():
        print("Starting Grid Search for hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")

        #4. Evaluate Model
        evaluate_model(best_model, X_test, y_test)

        #5. Log Feature Importance
        log_feature_importance(best_model)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            registered_model_name="BankLoanClassifier",
            code_paths=["src/bank_loan_cleaner.py", "src/feature_engineer.py"],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )  # Log the best model found during GridSearchCV


if __name__ == "__main__":
    main()