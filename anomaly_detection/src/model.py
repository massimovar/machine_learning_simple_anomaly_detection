def build_keras_model(input_size: int = 10, hidden_size: int = 64, bottleneck_size: int = 16):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_size,), name="sensor_input")
    encoded = tf.keras.layers.Dense(hidden_size, activation="relu")(inputs)
    encoded = tf.keras.layers.Dense(bottleneck_size, activation="relu")(encoded)
    decoded = tf.keras.layers.Dense(hidden_size, activation="relu")(encoded)
    outputs = tf.keras.layers.Dense(input_size, activation="sigmoid")(decoded)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")


def convert_to_tflite(keras_model, tflite_path: str) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    with open(tflite_path, "wb") as f:
        f.write(converter.convert())


def load_tflite_model(model_path: str, use_npu: bool = False):
    try:
        import tflite_runtime.interpreter as tflite

        Interpreter = tflite.Interpreter
        load_delegate = tflite.load_delegate
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
        load_delegate = tf.lite.experimental.load_delegate

    experimental_delegates = []
    if use_npu:
        try:
            experimental_delegates.append(load_delegate("libvx_delegate.so"))
        except Exception as exc:
            import logging

            logging.getLogger("anomaly-detector").warning(
                "NPU delegate unavailable; using CPU: %s", exc
            )

    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=experimental_delegates if experimental_delegates else None,
    )
    interpreter.allocate_tensors()
    return interpreter
