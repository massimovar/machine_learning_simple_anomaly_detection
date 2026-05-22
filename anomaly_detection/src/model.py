"""
model.py — Autoencoder definition and TFLite helpers
=====================================================

WHY THIS FILE EXISTS:
    This file defines the neural network at the heart of our anomaly detector.
    We use a simple "dense autoencoder" — a network that learns to compress and
    reconstruct its input.  If the reconstruction is bad (high error), the input
    is probably abnormal.

WHAT IS AN AUTOENCODER? (quick refresher)
    Imagine you have 10 sensor readings.  The encoder squeezes them down to just
    16 numbers (the "bottleneck").  The decoder then tries to rebuild the original
    10 numbers from those 16.  During training on NORMAL data, the network gets
    very good at this compression+reconstruction.  At inference time, an ABNORMAL
    sample looks different from anything it learned, so the reconstruction is poor
    and the error is high — that's our anomaly signal.

ARCHITECTURE:
    Input (10) → Dense(64) + ReLU → Dense(16) + ReLU   ← encoder
    Dense(16)  → Dense(64) + ReLU → Dense(10) + Sigmoid ← decoder

    - ReLU (Rectified Linear Unit): max(0, x).  A simple activation that lets
      the network learn non-linear patterns.
    - Sigmoid: squashes output to [0, 1], matching our MinMaxScaler-normalised inputs.

FRAMEWORK — TensorFlow Lite:
    Training is done with full TensorFlow/Keras on a development machine (x86_64).
    The trained Keras model is then converted to a flat .tflite file and deployed
    to the edge device.

    At inference time only tflite-runtime is needed — a very small package (~few MB)
    that can run .tflite models on CPU.

    TARGET DEVICE — NXP i.MX 8M Plus (ARM Cortex-A53):
    The edge device carries an integrated Neural Processing Unit (NPU).
    TFLite is the framework of choice in this project because it integrates
    with the NXP eIQ / VSI delegate stack used on the target device.
    NPU acceleration is not yet enabled in this version; the code is structured
    so that adding the delegate (see load_tflite_model) requires only a one-line
    change once the NPU driver is available on the device.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Model builder — used during training (requires full TensorFlow)
# ---------------------------------------------------------------------------

def build_keras_model(input_size: int = 10,
                      hidden_size: int = 64,
                      bottleneck_size: int = 16):
    """
    Build and return a Keras autoencoder model.

    As a functional Keras model (Input → layers → Output) this is easy to
    inspect, summarise, and later convert to TFLite.

    Parameters
    ----------
    input_size : int
        Number of input features (= number of sensor tags, typically 10).
    hidden_size : int
        Neurons in the first encoder layer and last decoder layer (default 64).
    bottleneck_size : int
        Neurons in the compressed middle layer (default 16).

    Returns
    -------
    tf.keras.Model
    """
    import tensorflow as tf  # imported lazily — not needed at inference time

    inputs = tf.keras.Input(shape=(input_size,), name="sensor_input")

    # --- Encoder --------------------------------------------------------------
    h1 = tf.keras.layers.Dense(hidden_size, activation="relu", name="enc_dense1")(inputs)
    bottleneck = tf.keras.layers.Dense(bottleneck_size, activation="relu", name="bottleneck")(h1)

    # --- Decoder --------------------------------------------------------------
    h2 = tf.keras.layers.Dense(hidden_size, activation="relu", name="dec_dense1")(bottleneck)
    # WHY Sigmoid: our inputs are normalised to [0, 1] by MinMaxScaler,
    # so we want the output in the same range.
    outputs = tf.keras.layers.Dense(input_size, activation="sigmoid", name="reconstruction")(h2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    return model


def convert_to_tflite(keras_model, tflite_path: str) -> None:
    """
    Convert a trained Keras model to a flat TFLite file and save it to disk.

    The resulting .tflite file is self-contained: it embeds the graph topology
    and the trained weights.  It can be run on any device that has tflite-runtime,
    including the NXP i.MX 8M Plus target.

    Parameters
    ----------
    keras_model : tf.keras.Model
        A trained model returned by build_keras_model().
    tflite_path : str
        Destination file path, e.g. "models/autoencoder.tflite".
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # Default conversion: 32-bit float, CPU-compatible.
    # For future NPU acceleration on the i.MX 8M Plus, enable post-training
    # quantisation here (e.g. converter.optimizations = [tf.lite.Optimize.DEFAULT])
    # and supply a representative dataset so the converter can determine
    # INT8 scale/zero-point values that the NPU delegate expects.
    tflite_bytes = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)


# ---------------------------------------------------------------------------
# Inference helpers — used at runtime (requires only tflite-runtime)
# ---------------------------------------------------------------------------

def load_tflite_model(model_path: str, use_npu: bool = False):
    """
    Load a .tflite model file and return an allocated TFLite Interpreter.

    The function first tries to import the lightweight ``tflite_runtime``
    package (installed on the edge device).  If that is not available it
    falls back to ``tensorflow.lite`` (available on the development machine).

    NPU DELEGATE (i.MX 8M Plus):
        ``use_npu=True`` will attempt to attach the NXP VSI NPU delegate.
        This requires the NXP eIQ runtime libraries to be present on the
        device (``libvx_delegate.so``).  Leave it False until the driver is
        deployed; the model will run on the ARM CPU in the meantime with no
        other code change needed.

    Parameters
    ----------
    model_path : str
        Path to the .tflite file.
    use_npu : bool
        If True, load the NXP VSI NPU delegate for hardware acceleration.
        Default False (CPU execution).

    Returns
    -------
    interpreter : tflite.Interpreter (allocated)
    """
    # --- Choose the interpreter backend ----------------------------------------
    try:
        # tflite-runtime: the small (~few MB) package installed on the edge device
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
        load_delegate = tflite.load_delegate
    except ImportError:
        # Fall back to the full TensorFlow package (development machine)
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        load_delegate = tf.lite.experimental.load_delegate

    # --- Optional NPU delegate (NXP i.MX 8M Plus eIQ VSI NPU delegate) --------
    experimental_delegates = []
    if use_npu:
        # The NXP VSI NPU delegate shared library is installed on the target
        # device by the eIQ runtime package.  The delegate accelerates all
        # operators it supports; unsupported ops fall back to CPU automatically.
        try:
            npu_delegate = load_delegate("libvx_delegate.so")
            experimental_delegates.append(npu_delegate)
        except Exception as exc:
            import logging
            logging.getLogger("anomaly-detector").warning(
                f"NPU delegate not available — falling back to CPU: {exc}"
            )

    # --- Build and allocate the interpreter ------------------------------------
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=experimental_delegates if experimental_delegates else None,
    )
    interpreter.allocate_tensors()
    return interpreter


# ---------------------------------------------------------------------------
# Legacy alias — kept so old call-sites don't break during refactoring
# ---------------------------------------------------------------------------

def load_model(model_path: str, input_size: int = 10,
               hidden_size: int = 64, bottleneck_size: int = 16,
               use_npu: bool = False):
    """
    Thin wrapper around load_tflite_model for backwards compatibility.

    Parameters
    ----------
    model_path : str
        Path to the .tflite file (previously .pth).
    use_npu : bool
        Passed through to load_tflite_model (default False — CPU only).

    Returns
    -------
    tflite.Interpreter
        An allocated TFLite Interpreter ready for inference.
    """
    # Kept for backward compatibility with older call-sites that passed these.
    _ = (input_size, hidden_size, bottleneck_size)
    return load_tflite_model(model_path, use_npu=use_npu)
