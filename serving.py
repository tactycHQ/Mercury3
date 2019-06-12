import tensorflow as tf

keras_model = tf.keras.models.load_model(".\saved_models\\run20.h5")
export_path = ".\saved_models\\1"

tf.saved_model.save(
        keras_model,
        export_path,
        signatures=None
)