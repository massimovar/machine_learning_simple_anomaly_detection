# Introduction

This demo is an **FTOptix** application that represents, in a simplified way, a digital twin of a staple production machine.

The goal of this demo is to detect anomalies directly on **OptixEdge**.

The objective is to unlock the value of field data, which is often collected but underutilized.

# The FTOptix Application

The FTOptix application is intentionally minimal:

- 10 sensors
- 1 button
- 1 feedback indicator

By clicking the button, we can generate an anomaly.

We can also interact with the 3D scene and explore useful information about the demo architecture (available in the top-right corner).

From a more technical perspective, the objective is to detect variations in **multivariate patterns** using machine learning.

This means identifying anomalies even when the values of individual sensors appear normal, including cases where they remain within configured alarm thresholds. The key is to analyze the relationships between variables, something that a machine learning model can do extremely well.

# Architecture

- FTOptix acquires data from the field and publishes it via MQTT to a machine learning model.
- The machine learning model performs inference on the incoming data and produces an output (a single bit) indicating whether an anomaly is present.

The MQTT broker and the machine learning model both run directly on **OptixEdge** inside two separate containers.

# The Machine Learning Model

The model used is an **autoencoder**.

It has been trained with thousands of samples of "good" data and therefore learns how to encode and decode only normal operating conditions.

At runtime, when it receives normal data:

1. It extracts the relevant features (the decoding phase), which correspond to the multivariate patterns and relationships among sensor values.
2. It then attempts to reconstruct the original data (the encoding phase).

## A Simple Analogy

Imagine the model has learned to process circles.

It can receive a circle, extract its key characteristics, and reconstruct it accurately:

**INPUT circle → decode → encode → OUTPUT circle**

Result: the output is very similar to the input.

Now imagine it receives a square instead of a circle.

Since it only knows how to reconstruct circles, the resulting output will be a poorly reconstructed circle, which differs significantly from the original square.

**INPUT square → decode → encode → OUTPUT circle**

Result: the output is very different from the input.

Therefore, if the difference between the **OUTPUT** and the **INPUT** exceeds a predefined threshold, the condition is classified as an anomaly.

# Conclusion and Key Takeaways

- FTOptix, the MQTT broker, and the machine learning model all run locally on OptixEdge, completely offline. Your data remains under your control.
- Extremely low latency thanks to local processing.
- Need to send data to the cloud? You do not need to transmit your raw data. In many cases, only the result of the analysis needs to be sent, such as the anomaly detection bit. This significantly reduces bandwidth requirements.

With this approach, there is no need to rely on traditional HMI alarms to identify complex anomalous patterns, a task that is often difficult and unreliable.

The most effective solution is to delegate this type of analysis to a machine learning model.

Equally important, machines evolve over time. Component wear, maintenance activities, and part replacements can create a "new normal" operating condition. The same machine learning model can adapt to these changes through retraining, allowing it to continue detecting anomalies accurately as the machine ages.