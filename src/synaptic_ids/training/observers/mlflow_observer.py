from typing import Any

import mlflow
from src.synaptic_ids.training.observers.observer import TrainingObserver


class MLflowObserver(TrainingObserver):
    def on_run_start(self, run_name: str, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        print(
            f"--- MLflow Observer: Run '{run_name}' started in experiment '{experiment_name}' ---"
        )

    def on_parameters_logged(self, params: dict):
        mlflow.log_params(params)
        print("MLflow Observer: Parameters logged.")

    def on_metrics_logged(self, metrics: dict):
        mlflow.log_metrics(metrics)
        print("MLflow Observer: Metrics logged.")

    def on_artifact_logged(self, local_path: str, artifact_path: str):
        mlflow.log_artifact(local_path, artifact_path)
        print(f"MLflow Observer: Artifact from {local_path} logged.")

    def on_model_logged(
        self,
        model,
        name: str,
        registered_model_name: str,
        input_example: Any = None,
        pip_requirements_path: Any = None,
    ):
        # If user passes a TensorFlow/Keras model directly, use the TF flavor.
        # If it's a custom PythonModel (like our SynapticIDSPipeline), use the pyfunc flavor.
        if isinstance(model, mlflow.pyfunc.PythonModel):
            mlflow.pyfunc.log_model(
                name=name,
                python_model=model,
                input_example=input_example,
                pip_requirements=pip_requirements_path,
                registered_model_name=registered_model_name,
            )
        else:
            mlflow.tensorflow.log_model(
                model=model,
                name=name,
                registered_model_name=registered_model_name,
                input_example=input_example,
                pip_requirements=pip_requirements_path,
            )
        print("MLflow Observer: Model logged.")

    def on_run_end(self):
        mlflow.end_run()
        print("MLflow Observer: Run ended.")
