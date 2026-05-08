import keras
from keras.layers import Dense, BatchNormalization

@keras.saving.register_keras_serializable(package="Custom")
class FixedDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

@keras.saving.register_keras_serializable(package="Custom")
class FixedBatchNormalization(BatchNormalization):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

try:
    model = keras.models.load_model(
        'final-saved-models/best_colon_cancer_efficientnet_transfer_learning.keras',
        custom_objects={'Dense': FixedDense, 'BatchNormalization': FixedBatchNormalization}
    )
    print("Success!")
    model.summary()
except Exception as e:
    print(f"Failed: {e}")
