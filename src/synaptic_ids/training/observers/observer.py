from abc import ABC, abstractmethod
from typing import Any, Dict


class TrainingObserver(ABC):
    @abstractmethod
    def on_run_start(self, run_name: str, experiment_name: str):
        pass

    @abstractmethod
    def on_parameters_logged(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def on_metrics_logged(self, metrics: Dict[str, float]):
        pass

    @abstractmethod
    def on_artifact_logged(self, local_path: str, artifact_path: str):
        pass

    @abstractmethod
    def on_model_logged(
        self,
        model,
        name: str,
        registered_model_name: str,
        input_example: Any = None,
        pip_requirements_path: str = None,
    ):
        pass

    @abstractmethod
    def on_run_end(self):
        pass
