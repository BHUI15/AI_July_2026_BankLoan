import mlflow
from mlflow.tracking import MlflowClient


# MLflow has a few distinct components. "Tracking" is where we record our experiments
# (like a lab notebook). The "Model Registry" is where we store our finalized,
# successful models (like a version-controlled catalog, similar to Git for code).
def register_and_promote_best_model(experiment_name: str, model_registry_name: str):
    """
    Finds the best model from an experiment, registers it, and promotes it to the "Production" stage.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        model_registry_name (str): The name under which to register the model in the Model Registry.
    """
    # MlflowClient is our "remote control" to talk to the MLflow server programmatically
    # without needing the UI.
    client = MlflowClient()

    # Get the experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")

    # Get all runs for the experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])

    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'.")

    # Get the best run (highest accuracy)
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    print(f"Best run ID: {best_run_id} with accuracy: {best_run.data.metrics['accuracy']}")

    # Register the model from the best run
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
    print(f"Model registered successfully with name: {registered_model.name} and version: {registered_model.version}")

    # Promote the registered model to "Production"
    client.transition_model_version_stage(
        name=registered_model.name,
        version=registered_model.version,
        stage="Production",
        archive_existing_versions=True  # Archive any existing production versions
    )
    print(f"Model version {registered_model.version} promoted to 'Production' stage.")


if __name__ == "__main__":
    # Example usage
    experiment_name = "Bank_Loan_Classification"
    model_registry_name = "BankLoanClassifier"
    register_and_promote_best_model(experiment_name, model_registry_name)