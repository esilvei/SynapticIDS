import mlflow
import pandas as pd

from src.synaptic_ids.processing.data_transformer.data_preparer import DataPreparer


class SynapticIDSPipeline(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow model that wraps the entire inference pipeline.

    This class ensures that the exact same preprocessing steps used during
    training are applied during inference. It takes raw data as input and
    outputs the final prediction.

    Attributes:
        model: The trained Keras model.
        data_preparer: The fitted DataPreparer object, containing the scaler
                       and all transformation logic.
    """
    def __init__(self, model, data_preparer: DataPreparer):
        self.model = model
        self.data_preparer = data_preparer

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full preprocessing and prediction pipeline.

        This method is called by `mlflow.pyfunc.load_model().predict()`.

        Args:
            context: The MLflow context (provided automatically).
            model_input: A Pandas DataFrame containing the raw input data,
                         matching the Pydantic schema.

        Returns:
            A DataFrame or NumPy array with the model's predictions.
        """
        # The preprocessing logic is now self-contained within the model artifact,
        # using the `data_preparer` that was fitted on the training data.
        prepared_data = self.data_preparer.prepare_data(model_input.copy(), is_training=False)

        # The output of prepare_data is a dictionary; we need the list of inputs
        # that the Keras model expects.
        model_inputs_list = [prepared_data["images"], prepared_data["sequences"]]

        # Perform inference with the trained Keras model
        return self.model.predict(model_inputs_list)