# The model-serving image. Continuous Delivery (.gitea/workflows/build-image.yaml) builds
# this on the runner and pushes it to the registry; your lab VM pulls and runs it with
# podman. It installs MLflow, copies the committed model_dir, installs the model's own
# declared dependencies, then serves it over HTTP with `mlflow models serve` — gunicorn
# only (no nginx), bound to 0.0.0.0 so the lab port map can reach it.

# 1 use the official lightwight Python image.
FROM python:3.13.5-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2 MLflow writes a local tracking/tracing SQLite database at serve time - this dir must exist and be writable.
WORKDIR /opt/mlflow

RUN pip install mlflow==3.14
# C:\Users\hbcbo\Desktop\AI_L6_Apprenticeship\module_7_project\AI_July_2026_BankLoan\mlruns\1\models\m-03a2d7610c284baea378f32d8a62f8e9\artifacts\MLmodel
COPY mlruns/1/models/m-03a2d7610c284baea378f32d8a62f8e9/artifacts /opt/ml/model
RUN python -c "from mlflow.models import container as C; C._install_pyfunc_deps('/opt/ml/model', install_mlflow=False, env_manager='local');"

ENTRYPOINT ["mlflow", "models", "serve", "-m", "/opt/ml/model", \
     "--host", "0.0.0.0", "--port", "8080", "--env-manager", "local"]