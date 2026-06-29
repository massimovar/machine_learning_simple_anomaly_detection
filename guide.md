# Guide to the Paper-Clip Machine Anomaly Detection Project

This guide is written for someone who is seeing this project for the first time. It explains the purpose, the vocabulary, the machine learning idea, the data flow, the Python files, the configuration, the training process, the runtime process, the MQTT messages, the Docker deployment, and the tests.

The text is intentionally prose-heavy because it may be converted to speech. File names and commands are included, but the main explanations are written as sentences rather than dense tables.

## 1. The project in one sentence

This project monitors a simulated paper-clip manufacturing machine, learns what normal sensor behavior looks like, and publishes an anomaly result whenever a recent window of sensor readings looks unusual.

In plain language, imagine a small industrial machine that feeds wire, bends the wire into a paper-clip shape, cuts it, and rejects bad parts. The machine has sensors. Those sensors produce values such as wire feed speed, motor current, vibration, temperature, and reject count. The project receives those values, groups them into short time windows, sends them through a very small machine learning model, and decides whether the current behavior is normal or anomalous.

The machine learning model is intentionally simple. It is an autoencoder. An autoencoder is a neural network trained to copy its input to its output. That sounds strange at first, but it is useful because the model is trained only on normal data. If the machine behaves normally, the autoencoder can reconstruct the input well. If the machine behaves abnormally, the autoencoder reconstructs poorly. The reconstruction error becomes the anomaly score.

## 2. The most important vocabulary

Before reading the code, it helps to learn the project language. These names are also recorded in `CONTEXT.md`.

A Paper-clip Machine is the simulated industrial asset being monitored. It has wire feeding, bending, cutting, vibration, temperature, and rejection behavior.

A Sensor Tag is one named measurement from the Paper-clip Machine. Examples are `wire_feed_speed`, `wire_tension`, `motor_current_feed`, `bend_angle`, and `temperature_motor`.

A Sensor Reading is one complete set of Sensor Tag values at one moment in time. In this project, the detector expects one Sensor Reading per second.

A Sensor Topic is the MQTT topic where incoming Sensor Readings arrive. In the default configuration, it is `ftoptix/paperclip/sensors`.

An Alert Topic is the MQTT topic where outgoing Detection Results are published. In the default configuration, it is `anomaly/paperclip/alerts`.

A Detection Window is a fixed-size group of consecutive Sensor Readings. The default window size is thirty readings. At one reading per second, this means one Detection Window covers thirty seconds.

A Feature Vector is the numeric model input created from a Detection Window. The project averages each Sensor Tag over the window, so thirty raw readings become one vector of ten numbers.

Normal Training Data is data that represents expected Paper-clip Machine behavior. The autoencoder trains only on normal data.

An Anomalous Reading is a reading whose values are outside expected behavior. In the synthetic test data, anomalies are created by pushing one to three Sensor Tags far outside their normal ranges.

An Autoencoder is the neural network that learns to reconstruct normal Feature Vectors.

Reconstruction Error is the mean squared difference between the input Feature Vector and the autoencoder output. Low Reconstruction Error means the model reconstructed the input well. High Reconstruction Error means the model struggled.

An Anomaly Threshold is the decision value saved after training. If the Reconstruction Error is greater than the Anomaly Threshold, the Detection Result is anomalous.

A Detection Result is the output of one completed Detection Window. It contains `is_anomaly`, `anomaly_score`, and `threshold`.

Training Artifacts are the files needed at runtime: the TFLite autoencoder file, the fitted scaler, and the threshold JSON file.

The Edge Detector is the Python runtime that listens to MQTT, preprocesses Sensor Readings, runs the model, and publishes Detection Results.

## 3. The project folders and files

The main exercise lives in the `anomaly_detection` folder.

The central configuration file is `anomaly_detection\config.yaml`. It defines MQTT topics, preprocessing settings, model paths, and training parameters.

The runtime Python code lives in `anomaly_detection\src`.

The file `anomaly_detection\src\main.py` is the entry point. It loads configuration, loads Training Artifacts, connects to the MQTT broker, receives Sensor Readings, runs detection, logs the result, and publishes the Detection Result.

The file `anomaly_detection\src\mqtt_client.py` wraps the `paho-mqtt` library. It creates the MQTT client, subscribes to the Sensor Topic, parses incoming JSON, flattens the FTOptix Records Envelope when needed, and publishes alert JSON to the Alert Topic.

The file `anomaly_detection\src\preprocessing.py` owns the transformation from raw Sensor Readings to normalised Feature Vectors. It validates Sensor Tags, buffers readings into Detection Windows, averages each window, and normalises the result with the saved scaler.

The file `anomaly_detection\src\model.py` defines the Keras autoencoder used during training and also contains the TFLite loading helpers used at runtime.

The file `anomaly_detection\src\detector.py` runs the TFLite model on one Feature Vector, computes the Reconstruction Error, compares it with the Anomaly Threshold, and returns a Detection Result.

The training code lives in `anomaly_detection\training`.

The file `anomaly_detection\training\generate_synthetic.py` creates fake paper-clip machine data. It writes normal training data and mixed test data.

The file `anomaly_detection\training\train.py` trains the autoencoder on normal data, fits the scaler, computes the threshold, converts the trained model to TFLite, and saves the Training Artifacts.

The tests live in `anomaly_detection\tests`.

The file `anomaly_detection\tests\test_preprocessing.py` tests the preprocessing module without MQTT or Docker.

The file `anomaly_detection\tests\test_integration.py` tests the full running stack through MQTT. It publishes normal and anomalous windows and checks that the detector publishes the expected Detection Results.

The deployment files are in the root of `anomaly_detection`.

The file `anomaly_detection\Dockerfile` builds the detector container for ARM64 edge deployment. It installs `tflite-runtime`, copies the configuration, source code, and model artifacts, then runs `python src/main.py`.

The file `anomaly_detection\docker-compose.yml` defines two services: a Mosquitto MQTT broker and the detector container.

The file `anomaly_detection\mosquitto.conf` is a simple Mosquitto configuration for port 1883 with anonymous access enabled for the demo.

The file `anomaly_detection\requirements.txt` lists the Python runtime dependencies that are not TensorFlow. TensorFlow is intentionally not listed there because full TensorFlow is needed for local training, while `tflite-runtime` is used for lightweight edge inference.

## 4. The ten Sensor Tags

The detector expects exactly ten Sensor Tags, in a fixed order. The order matters because the autoencoder sees a numeric vector, not named values.

The configured Sensor Tags are:

First, `wire_feed_speed`, which is the wire feed speed in meters per minute.

Second, `wire_tension`, which is the wire tension in newtons.

Third, `motor_current_feed`, which is the feed motor current in amperes.

Fourth, `motor_current_bend`, which is the bending motor current in amperes.

Fifth, `bend_angle`, which is the bend angle in degrees.

Sixth, `cycle_time`, which is the cycle time in milliseconds.

Seventh, `vibration_rms`, which is the root mean square vibration level.

Eighth, `cut_force`, which is the cutting force in newtons.

Ninth, `temperature_motor`, which is the motor temperature.

Tenth, `reject_count_per_min`, which is the number of rejected parts per minute.

The synthetic data generator uses plausible normal ranges for these tags. For example, wire feed speed is centered around ten meters per minute, bend angle is centered around one hundred eighty degrees, and cycle time is centered around two hundred milliseconds.

## 5. How MQTT fits into the project

MQTT is a lightweight publish and subscribe messaging protocol. In this project, it acts like a message post office.

FTOptix, or a test publisher, publishes Sensor Readings to the Sensor Topic. The Mosquitto broker receives those messages. The Edge Detector subscribes to the Sensor Topic, so the broker delivers each Sensor Reading to the detector. After processing a complete Detection Window, the Edge Detector publishes a Detection Result to the Alert Topic. FTOptix, a test subscriber, or another consumer can subscribe to the Alert Topic.

The default Sensor Topic is `ftoptix/paperclip/sensors`.

The default Alert Topic is `anomaly/paperclip/alerts`.

Incoming data can arrive in two accepted shapes.

The first accepted shape is flat JSON. In flat JSON, the payload is simply a dictionary where each key is a Sensor Tag and each value is numeric.

The second accepted shape is the FTOptix Records Envelope. In that format, the payload contains a `Records` array. Each record has a `TagName` and a `Value`. The MQTT module converts that envelope into the same flat dictionary used by the rest of the detector.

This conversion is important because it keeps the rest of the project independent from FTOptix's exact JSON envelope. After `mqtt_client.py` has parsed the message, preprocessing only sees a plain Sensor Reading dictionary.

## 6. The runtime data flow from MQTT to anomaly result

This is the runtime path when the Edge Detector is running.

First, `main.py` opens `config.yaml`.

Second, it reads the MQTT configuration, preprocessing configuration, and model configuration.

Third, it loads the Training Artifacts. The TFLite autoencoder is loaded from `models\autoencoder.tflite`. The scaler is loaded from `models\scaler.pkl`. The Anomaly Threshold is loaded from `models\threshold.json`.

Fourth, it creates a `SampleBuffer`. This object knows the window size and the Sensor Tag order. The scaler is loaded separately.

Fifth, it creates an MQTT client with `create_client` from `mqtt_client.py`. That client connects to the broker and subscribes to the Sensor Topic.

Sixth, every time a Sensor Reading arrives, `mqtt_client.py` parses the JSON and calls the nested `handle_sensor_message` function in `main.py`.

Seventh, `handle_sensor_message` passes the Sensor Reading to `SampleBuffer.add`.

Eighth, if the Detection Window is not full yet, `SampleBuffer.add` returns `None`. The detector waits for more readings.

Ninth, when the Detection Window reaches thirty readings, `main.py` averages the window with `aggregate_window`, normalises it with `normalise`, and resets the buffer for the next window.

Tenth, `main.py` calls `detect` from `detector.py`.

Eleventh, `detect` writes the Feature Vector into the TFLite interpreter, invokes the model, reads the reconstructed output, computes the Reconstruction Error, and compares it with the Anomaly Threshold.

Twelfth, `detect` returns a Detection Result.

Thirteenth, `main.py` logs either normal or anomaly, converts the Detection Result to JSON, and publishes it to the Alert Topic with `publish_alert`.

The detector publishes every Detection Result, not only anomalous ones. This is deliberate. A consumer can see continuous status and can filter on `is_anomaly` if it only cares about anomalies.

## 7. Why preprocessing exists

Raw sensor data is not sent directly into the autoencoder. It must first be made stable, ordered, and scaled.

The first preprocessing task is validation. The detector expects all configured Sensor Tags to be present. If a Sensor Reading is missing a required tag, `PreprocessingError` is raised. If a tag value cannot be converted to a number, `PreprocessingError` is raised. This is better than allowing a cryptic model error later.

The second preprocessing task is ordering. JSON dictionaries have names, but neural networks receive arrays. The model expects the first number to mean wire feed speed, the second number to mean wire tension, and so on. The configured `tag_names` list defines this order.

The third preprocessing task is windowing. One raw Sensor Reading may contain noise. Instead of reacting to every single second, the detector groups thirty readings into one Detection Window.

The fourth preprocessing task is aggregation. The project averages each Sensor Tag over the Detection Window. This creates one ten-number Feature Vector from thirty ten-number readings.

The fifth preprocessing task is normalisation. The autoencoder was trained on values scaled to the range from zero to one. Runtime data must use the same transformation. That transformation is stored in `models\scaler.pkl`.

This preprocessing keeps the machine learning exercise simple. The autoencoder always receives one fixed-size vector of ten normalised numbers.

## 8. Why the scaler is necessary

The Sensor Tags use different units and different numeric ranges.

For example, vibration might be around one or two, motor current might be around two or three, cycle time might be around two hundred, and bend angle might be around one hundred eighty.

If the model used raw values, large numeric ranges could dominate the error calculation. Cycle time would influence the model more than vibration simply because the numbers are bigger, not because it is more important.

The MinMaxScaler solves this by mapping each Sensor Tag into a comparable zero-to-one range. It learns the minimum and maximum values from Normal Training Data, then applies the same mapping at runtime.

This is why training and runtime must use the same scaler. If you train with one scaler but run inference with a different scaler, the model receives numbers in a different coordinate system and the anomaly score becomes unreliable.

## 9. The autoencoder idea, step by step

The autoencoder has one job: reconstruct normal Feature Vectors.

The input has ten numbers because there are ten Sensor Tags.

The first dense layer expands or transforms those ten numbers into sixty-four hidden values.

The bottleneck layer compresses the representation into sixteen values.

The decoder then expands back to sixty-four values.

The final layer outputs ten numbers, one reconstructed value for each original input value.

The model architecture is: input size ten, hidden size sixty-four, bottleneck size sixteen, hidden size sixty-four, output size ten.

The hidden layers use ReLU activation. ReLU stands for rectified linear unit. It helps the network learn non-linear patterns.

The final layer uses sigmoid activation. Sigmoid outputs values between zero and one, which matches the normalised input range.

During training, the model receives a normalised Feature Vector as input and is asked to output the same vector. The target equals the input. This is called a reconstruction task.

If the model sees many normal examples, it learns the patterns that are common in normal operation. It does not learn a labeled list of failures. It learns the shape of normality.

At runtime, a normal Feature Vector should be reconstructed accurately. An abnormal Feature Vector should be reconstructed less accurately. That difference is the signal used for anomaly detection.

## 10. Reconstruction Error and the Anomaly Threshold

The anomaly score is the Reconstruction Error.

The project computes Reconstruction Error with mean squared error. For each Sensor Tag, it subtracts the reconstructed value from the original normalised value, squares the difference, and then averages those squared differences across all ten tags.

A small mean squared error means the reconstruction is close to the input. That suggests normal behavior.

A large mean squared error means the reconstruction is far from the input. That suggests anomalous behavior.

The Anomaly Threshold is computed during training. After the autoencoder is trained, the training script reconstructs all normal training Feature Vectors and computes their reconstruction errors. Then it calculates the average error and the standard deviation of those errors.

The default threshold rule is mean plus three times the standard deviation. This is often called the three-sigma rule.

The intuition is simple. If training errors are mostly small, then a new error much larger than the normal training range is suspicious. The threshold marks the line between expected reconstruction error and unusually high reconstruction error.

At runtime, the decision is direct. If anomaly score is greater than threshold, `is_anomaly` is true. Otherwise, `is_anomaly` is false.

## 11. How synthetic data is generated

The project does not require a real Paper-clip Machine. The script `training\generate_synthetic.py` creates fake but plausible data.

For each Sensor Tag, the generator defines a center value and a noise half-width.

For normal data, each value is the center plus random uniform noise. For example, if wire feed speed has center ten and noise two, normal values are approximately between eight and twelve.

The script creates `data\normal_train.csv` with five thousand normal readings. This is used for training.

The script also creates `data\test_mixed.csv` with one thousand readings. About ninety percent are normal, and about ten percent are anomalous. The anomalous readings are created by choosing one to three Sensor Tags and pushing them three to five times outside their normal noise range.

The mixed file includes an `is_anomaly` label column for evaluation. The normal training file does not include anomalies, because the autoencoder should learn only normal behavior.

To generate the synthetic data, run this command from the `anomaly_detection` folder:

```text
python training\generate_synthetic.py
```

## 12. How training works

Training happens in `training\train.py`.

First, the script loads `config.yaml`.

Second, it reads `data\normal_train.csv`.

Third, it simulates runtime windowing. Runtime detection receives one Sensor Reading at a time and averages every thirty readings. Training must match that, so the training script groups the normal CSV rows into non-overlapping windows of thirty rows and averages each window.

Fourth, it fits a MinMaxScaler on the averaged normal windows. This scaler learns how to map each Sensor Tag into the zero-to-one range.

Fifth, it builds the Keras autoencoder by calling `build_keras_model` from `src\model.py`.

Sixth, it compiles the model with the Adam optimizer and mean squared error loss.

Seventh, it trains the model. The configured default is one hundred epochs, learning rate zero point zero zero one, and batch size thirty-two.

Eighth, it reconstructs all scaled training windows.

Ninth, it computes the mean squared error for each reconstructed window.

Tenth, it computes the threshold using mean plus sigma times standard deviation. The default sigma value is three.

Eleventh, it saves the Training Artifacts.

The TFLite autoencoder is saved to `models\autoencoder.tflite`.

The fitted scaler is saved to `models\scaler.pkl`.

The threshold information is saved to `models\threshold.json`.

To train the model, run this command from the `anomaly_detection` folder:

```text
python training\train.py
```

For local training, you need full TensorFlow installed. TensorFlow is not listed in `requirements.txt` because the runtime edge container uses `tflite-runtime` instead.

## 13. How runtime inference works

Runtime inference is the process of using the trained model to score new data.

At runtime, the Edge Detector does not train. It only loads the saved Training Artifacts.

The autoencoder is loaded as a TFLite interpreter. TFLite is TensorFlow Lite, a smaller inference format designed for deployment.

The detector writes one normalised Feature Vector into the interpreter input tensor. TFLite expects a batch dimension, so the vector is reshaped from ten values into one row with ten values.

The detector calls `invoke` on the interpreter. That runs the autoencoder.

The detector reads the output tensor. That output is the reconstructed Feature Vector.

Then `detector.py` computes the Reconstruction Error by comparing the input Feature Vector to the reconstructed Feature Vector.

Finally, it compares the score with the Anomaly Threshold and creates a Detection Result.

## 14. The Detection Result JSON

The outgoing Detection Result is published as JSON.

It contains three fields.

The first field is `is_anomaly`. This is a boolean. It is true when the anomaly score is greater than the threshold.

The second field is `anomaly_score`. This is the Reconstruction Error rounded to six decimal places.

The third field is `threshold`. This is the Anomaly Threshold rounded to six decimal places.

An example Detection Result looks like this:

```text
{
  "is_anomaly": true,
  "anomaly_score": 0.042871,
  "threshold": 0.003412
}
```

If `is_anomaly` is false and the anomaly score is much lower than the threshold, the machine is behaving normally.

If `is_anomaly` is false but the score is close to the threshold, the behavior is still classified as normal, but it may be worth watching.

If `is_anomaly` is true and the score is much higher than the threshold, the detector believes the current Detection Window is clearly unusual.

## 15. The configuration file

The file `config.yaml` controls the main tunable behavior.

The `mqtt` section defines the broker host, broker port, Sensor Topic, Alert Topic, and MQTT client identifier. Inside Docker Compose, the broker host is `mqtt-broker`, because that is the service name of the Mosquitto container.

The `preprocessing` section defines the Detection Window size and the ordered Sensor Tag names. The default window size is thirty.

The `model` section defines the autoencoder input size, hidden size, bottleneck size, model path, scaler path, threshold path, and whether to try NPU acceleration.

The `training` section defines the number of epochs, learning rate, batch size, and threshold sigma.

If you change the Sensor Tag list or model input size, you must retrain the model. The saved autoencoder, scaler, and threshold all depend on that input shape.

If you change the window size, you should retrain the model. Training simulates the same windowing behavior used at runtime, so changing the window size changes the kind of Feature Vectors the model sees.

If you change only MQTT topics or broker host, you do not need to retrain.

## 16. Docker and edge deployment

The project can run locally on your development machine and on a low-power ARM64 edge device such as the NXP i.MX 8M Plus.

The Dockerfile uses `python:3.11-slim` and installs `tflite-runtime`. By default it builds for the local container platform. The `docker-compose.edge.yml` override pins the detector service to `linux/arm64` for edge builds.

The Docker image copies `config.yaml`, `src`, and `models` into `/app`. Then it starts the detector with `python src/main.py`.

The Compose file defines two services.

The first service is `mqtt-broker`. It uses the `eclipse-mosquitto:2` image. It exposes port 1883 for MQTT and port 9001 for WebSocket MQTT. It allows anonymous connections because this is a demo exercise.

The second service is `detector`. In the default Compose file it builds the local image `anomaly_detection-detector:local` for your current machine. For edge deployment, combine it with `docker-compose.edge.yml` to build/run `anomaly_detection-detector:edge` as `linux/arm64`. It depends on the broker health check.

The detector service does not publish its own port because it communicates through MQTT, not HTTP.

To run the stack, use the container tooling available in your environment. On Windows with Podman, `podman compose` works; if Podman is exposed through Docker-compatible commands, `docker compose` is equivalent. For example:

```text
podman compose up -d --build
```

Before running the detector container, make sure the `models` folder contains the Training Artifacts. If the model, scaler, or threshold files are missing, the detector cannot start correctly.

## 17. NPU acceleration

The code includes a switch for NPU acceleration, but it is disabled by default.

In `config.yaml`, the setting is `use_npu: false`.

If set to true, `model.py` tries to load the NXP VSI NPU delegate library named `libvx_delegate.so`. If the delegate is available, TFLite can offload supported operations to the NPU. If the delegate is not available, the code logs a warning and falls back to CPU.

For this simple exercise, CPU inference is enough. The model is tiny, and it processes only one Feature Vector every thirty seconds by default.

## 18. Tests

There are two levels of testing.

The first level is focused preprocessing testing. The file `tests\test_preprocessing.py` checks behavior without MQTT, Docker, or TensorFlow. It verifies Sensor Tag order, missing and non-numeric Sensor Tags, window averaging, invalid Detection Window shapes, and scaler-based normalisation.

Run the focused preprocessing tests from the `anomaly_detection` folder:

```text
python -m unittest discover -s tests -p test_preprocessing.py
```

The second level is integration testing. The file `tests\test_integration.py` assumes the full stack is already running. It creates a test alert receiver, publishes thirty normal Sensor Readings, waits for a Detection Result, and expects `is_anomaly` to be false. Then it publishes thirty anomalous Sensor Readings, waits for another Detection Result, and expects `is_anomaly` to be true.

Run the integration test from the `anomaly_detection` folder after starting the stack:

```text
python tests\test_integration.py
```

The integration test requires a running MQTT broker and detector container. If it times out, the most likely causes are that the stack is not running, the broker port is unavailable, the detector did not start, or the Training Artifacts are missing.

## 19. Common things to change during the exercise

If you want faster feedback, reduce the Detection Window size in `config.yaml`. Remember that changing the window size changes the training input, so retrain after changing it.

If you want the model to be stricter or more tolerant, change `threshold_sigma` in `config.yaml` and retrain. A smaller sigma makes anomalies easier to trigger but may increase false alarms. A larger sigma makes the detector more tolerant but may miss subtle anomalies.

If you want a larger or smaller neural network, change `hidden_size` or `bottleneck_size` and retrain. A larger network may reconstruct more patterns but may also learn too much. A smaller network may be simpler but may reconstruct normal behavior poorly.

If you want to add a new Sensor Tag, update `tag_names`, update the synthetic data generator, update the model input size, and retrain. The input order must stay consistent across data generation, training, preprocessing, and inference.

If you want to connect a different producer, publish JSON to the Sensor Topic using the same Sensor Tag names. The payload may be flat JSON or the FTOptix Records Envelope.

## 20. Common failure modes

If the detector says a Sensor Reading is missing a required Sensor Tag, check that the publisher is using the same tag names as `config.yaml`. The names must match exactly.

If the model file cannot be loaded, check that `models\autoencoder.tflite` exists and that the path in `config.yaml` is correct.

If the scaler cannot be loaded, check that `models\scaler.pkl` exists. The detector needs the scaler because runtime data must be normalised the same way as training data.

If the threshold cannot be loaded, check that `models\threshold.json` exists and contains a `threshold` field.

If the integration test receives no alert, check that the MQTT broker is running, port 1883 is reachable, the detector container is running, and the detector logs do not show startup errors.

If every reading is anomalous, check whether training and runtime preprocessing match. The most common causes are changed Sensor Tag order, changed window size without retraining, wrong scaler, or synthetic anomaly values being much farther outside the training range than expected.

If no reading is anomalous, check whether the threshold is too high, whether anomalous values are actually being published, and whether the detector is receiving the Sensor Tags in the expected names.

## 21. Why the architecture is intentionally simple

This is a learning project, not an industrial-grade anomaly detection platform.

The code avoids a large framework. It uses a few focused modules.

`mqtt_client.py` owns MQTT parsing and publishing.

`preprocessing.py` owns the contract from Sensor Reading to normalised Feature Vector.

`model.py` owns model creation, conversion, and loading.

`detector.py` owns scoring and threshold comparison.

`main.py` owns orchestration.

This separation matters because each file has one main reason to change. If the MQTT payload changes, look in `mqtt_client.py`. If Sensor Tag validation or windowing changes, look in `preprocessing.py`. If the model architecture changes, look in `model.py`. If the score calculation changes, look in `detector.py`. If startup wiring changes, look in `main.py`.

The deepest runtime object is now `SampleBuffer`. Its interface is small: add a Sensor Reading, and either get nothing yet or get one raw Detection Window. `main.py` then calls the explicit `aggregate_window` and `normalise` functions. This keeps the pipeline minimal and easy to follow.

## 22. The full lifecycle

The full project lifecycle is:

First, generate synthetic Sensor Readings.

Second, train the autoencoder on Normal Training Data.

Third, save the Training Artifacts.

Fourth, build or run the detector container.

Fifth, publish live or test Sensor Readings to the Sensor Topic.

Sixth, let the Edge Detector collect each Detection Window.

Seventh, let the autoencoder reconstruct the normalised Feature Vector.

Eighth, compute Reconstruction Error.

Ninth, compare Reconstruction Error with the Anomaly Threshold.

Tenth, publish the Detection Result to the Alert Topic.

If you understand those ten steps, you understand the core of the project.

## 23. A mental model to remember

Think of the project as a quality inspector with memory of normal behavior.

Training is the inspector watching many examples of the Paper-clip Machine working normally.

The scaler is the inspector learning how to compare different kinds of measurements fairly.

The autoencoder is the inspector learning to redraw normal behavior from memory.

The Reconstruction Error is the difference between what the inspector expected and what actually happened.

The Anomaly Threshold is the line where the difference becomes too large to ignore.

MQTT is the conveyor belt that brings readings to the inspector and carries results away.

The Edge Detector is the complete inspector running near the machine.

## 24. Minimal command sequence for a new developer

From the `anomaly_detection` folder, generate data:

```text
python training\generate_synthetic.py
```

Install local training requirements, including TensorFlow, if they are not already installed.

Train the model:

```text
python training\train.py
```

Run focused preprocessing tests:

```text
python -m unittest discover -s tests -p test_preprocessing.py
```

Start the stack with your container tool. For example:

```text
podman compose up -d --build
```

Run the integration test:

```text
python tests\test_integration.py
```

At that point, you have exercised the complete path from Sensor Readings through MQTT, preprocessing, autoencoder inference, Reconstruction Error scoring, threshold comparison, and alert publishing.
