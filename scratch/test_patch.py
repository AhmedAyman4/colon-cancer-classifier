import keras
from keras.layers import Dense, BatchNormalization, Dropout

# Monkey patch Dense to ignore quantization_config
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init

# Also for other layers that might have it
original_bn_init = BatchNormalization.__init__
def patched_bn_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return original_bn_init(self, *args, **kwargs)
BatchNormalization.__init__ = patched_bn_init

try:
    print("Loading model with monkey patches...")
    model = keras.models.load_model('final-saved-models/best_colon_cancer_efficientnet_transfer_learning.keras')
    print("Success!")
    model.summary()
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed: {e}")
