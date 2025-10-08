import asyncio

import mlflow
import numpy as np
import pandas as pd


class SynapticIDSPipeline(mlflow.pyfunc.PythonModel):
    """A custom MLflow model pipeline for SynapticIDS."""

    def __init__(self, model, data_preparer):
        """Initializes the pipeline with the trained model and data preparer."""
        self.model = model
        self.data_preparer = data_preparer

    async def preprocess(self, model_input):
        """Preprocesses the input data before prediction."""
        return await self.data_preparer.prepare_data(model_input)

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame
    ) -> np.ndarray:
        """Generates predictions for the given model input."""
        # Preprocess the data
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 'get_running_loop' fails in a new thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        processed_data = loop.run_until_complete(self.preprocess(model_input))

        # Extract images and sequences
        images = processed_data["images"]
        sequences = processed_data["sequences"]

        # Make predictions
        predictions = self.model.predict([images, sequences])

        # Post-process predictions
        return self.postprocess(predictions)

    def postprocess(self, predictions):
        """Post-processes the model's predictions."""
        # You can add more post-processing logic here if needed
        print(f"Raw predictions: {predictions}")
        return predictions
