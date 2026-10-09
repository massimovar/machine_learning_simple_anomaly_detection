# Edge AI on a Paper-Clip Machine — A Learning Tutorial

> **What is this?** A self-contained learning project with two small AI applications for an industrial HMI built with **FactoryTalk Optix**. Both target the **NXP i.MX 8M Plus** edge controller (ARM Cortex-A53 CPU with an on-chip Neural Processing Unit, NPU), run in containers and talk to the HMI over **MQTT**. There is no GPU and no cloud.
>
> - **Part 1, anomaly detection:** a lightweight neural network that we train ourselves watches the machine's sensor data and raises an alert when the pattern looks abnormal. Inference runs on **TensorFlow Lite** because it integrates with the NXP eIQ/VSI delegate stack used on this target platform. By default it runs on the ARM CPU.
> - **Part 2, gesture control (work in progress):** a camera watches the operator, a ready-made pose model (**MoveNet**) finds the body joints in every picture, and simple rules turn hand movements into four HMI commands. The service is built and tested on a laptop; the FactoryTalk Optix side and the NPU are not built yet ([status](#92-what-is-built-and-what-is-not)).
> - **Part 3, anomaly detection on the NPU:** the same detector, with an int8 model, running on the NPU from inside a container that borrows the device's NPU libraries read-only, with nothing installed on the device, and the proof shown on a FactoryTalk Optix panel ([status](#26-goal-and-ground-rules)).
>
> **No machine-learning background is assumed:** every concept is explained from scratch.

---

<img src="misc/screenshot.png" width="100%"/>

## Start Here

| I want to… | Go to |
|---|---|
| Understand the anomaly-detection project from zero | [1. Introduction](#1-introduction) |
| Run the anomaly detector | [7. How to Run](#7-how-to-run) |
| See what gesture control does and how finished it is | [9. Gesture Control Overview](#9-gesture-control-overview) |
| Learn the ML ideas behind it (keypoints, MoveNet, quantization, NPU) | [10. Machine Learning Basics for Gesture Control](#10-machine-learning-basics-for-gesture-control) |
| Understand how a hand movement becomes a command | [13. From Keypoints to Gestures](#13-from-keypoints-to-gestures) |
| Run the gesture service on a Windows laptop | [23. How to Run the Gesture Service](#23-how-to-run-the-gesture-service) |
| Know what is missing and what comes next | [25. Gesture Status and Next Steps](#25-gesture-status-and-next-steps) |
| Understand how the anomaly detector runs on the NPU, and how it is proven | [Part 3: 26–35](#26-goal-and-ground-rules) |
| Look up a term | [Glossary](#glossary) |
| Check where a fact comes from | [Sources](#sources) |

**New to all of this?** Read [2.1 What is MQTT?](#21-what-is-mqtt) first, then sections 9 to 15 in order, then try [section 23](#23-how-to-run-the-gesture-service).

## Table of Contents

**Part 1: Anomaly Detection**

1. [Introduction](#1-introduction)
2. [Background Concepts](#2-background-concepts)
3. [How the Algorithm Works](#3-how-the-algorithm-works)
4. [Architecture & Data Flow](#4-architecture--data-flow)
5. [Project Walkthrough](#5-project-walkthrough)
6. [Decision Log](#6-decision-log)
7. [How to Run](#7-how-to-run)
8. [Experimenting & Next Steps](#8-experimenting--next-steps)

**Part 2: Gesture Control**

9. [Gesture Control Overview](#9-gesture-control-overview)
10. [Machine Learning Basics for Gesture Control](#10-machine-learning-basics-for-gesture-control)
11. [Gesture Architecture](#11-gesture-architecture)
12. [From Camera Frame to Keypoints](#12-from-camera-frame-to-keypoints)
13. [From Keypoints to Gestures](#13-from-keypoints-to-gestures)
14. [One Gesture, One Event](#14-one-gesture-one-event)
15. [MQTT Messages for Gestures](#15-mqtt-messages-for-gestures)
16. [FactoryTalk Optix Integration (Planned)](#16-factorytalk-optix-integration-planned)
17. [Health Monitoring and Camera Loss](#17-health-monitoring-and-camera-loss)
18. [Containers and the Laptop Setup](#18-containers-and-the-laptop-setup)
19. [Performance and the NPU Plan](#19-performance-and-the-npu-plan)
20. [Gesture Code Walkthrough](#20-gesture-code-walkthrough)
21. [Testing the Gesture Service](#21-testing-the-gesture-service)
22. [Gesture Decision Log](#22-gesture-decision-log)
23. [How to Run the Gesture Service](#23-how-to-run-the-gesture-service)
24. [Tuning the Gesture Parameters](#24-tuning-the-gesture-parameters)
25. [Gesture Status and Next Steps](#25-gesture-status-and-next-steps)

**Part 3: Anomaly Detection on the NPU**

26. [Goal and Ground Rules](#26-goal-and-ground-rules)
27. [What an NPU Needs](#27-what-an-npu-needs)
28. [Quantization on Our Own Model](#28-quantization-on-our-own-model)
29. [The NPU-Aware Detector](#29-the-npu-aware-detector)
30. [Evidence over MQTT](#30-evidence-over-mqtt)
31. [Why the NPU May Be Slower Here](#31-why-the-npu-may-be-slower-here)
32. [The FactoryTalk Optix Proof Panel](#32-the-factorytalk-optix-proof-panel)
33. [Deploying to the Device](#33-deploying-to-the-device)
34. [The Ladder Log: Predictions vs Results](#34-the-ladder-log-predictions-vs-results)
35. [Decision Log](#35-decision-log)

**Reference**

- [Glossary](#glossary)
- [Sources](#sources)

---

# Part 1 — Anomaly Detection

> Sections 1 to 8: a small neural network learns what "normal" sensor data looks like and flags anything else.

## 1. Introduction

### What is this project?

This project monitors a **paper-clip manufacturing machine** — a line that feeds steel wire, bends it into paper-clip shape, cuts it, and packages it. Sensors on the machine produce readings (motor current, temperature, vibration, etc.) once per second. Our Python application subscribes to those readings over MQTT, feeds them to a small neural network called an **autoencoder**, and raises an alert when the sensor pattern looks abnormal.

### Who is it for?

Developers and engineers who want to understand:

- How machine learning can detect equipment problems in real time
- How MQTT works for industrial data
- How to package an ML application in Docker for edge deployment

**Prerequisite**: Basic Python knowledge. No ML or IoT experience required — this README teaches everything from scratch.

### What will you learn?

- **MQTT**: publish/subscribe messaging for IoT
- **Anomaly detection**: finding unusual patterns without labelled examples
- **Autoencoders**: a type of neural network that learns "normal" and flags deviations
- **Data preprocessing**: windowing, averaging, and normalisation
- **Edge deployment**: running ML in a Docker container on a small device

---

## 2. Background Concepts

### 2.1 What is MQTT?

**MQTT** (Message Queuing Telemetry Transport) is a lightweight messaging protocol designed for devices with limited power and bandwidth — exactly the kind of devices found in factories.

It uses a **publish/subscribe** pattern:

```
                         ┌───────────────┐
  FTOptix (publisher) ──►│  MQTT Broker  │──► Anomaly Detector (subscriber)
                         └───────────────┘
```

- **Publisher**: a device or application that *sends* messages to a **topic** (like a channel name).
- **Broker**: a server that *receives* messages and *forwards* them to anyone subscribed to that topic.
- **Subscriber**: an application that *listens* on a topic and receives messages as they arrive.

**Why MQTT for industrial IoT?**

| Feature | Why it matters |
|---|---|
| Tiny overhead | Works on constrained devices and slow networks |
| Decoupled | Publisher doesn't need to know who's listening |
| Reliable | Supports QoS levels (0 = fire-and-forget, 1 = at least once, 2 = exactly once) |

In our project, **Rockwell Automation FTOptix** publishes sensor data to the topic `ftoptix/paperclip/sensors`. Our detector subscribes to that topic and processes each message.

### 2.2 What is Anomaly Detection?

Anomaly detection answers the question: **"Is this data point normal or unusual?"**

**Real-world examples:**

- A credit card company flags a transaction in a country you've never visited → anomaly
- A machine's motor current spikes 3× higher than usual → anomaly
- A server suddenly starts using 100% CPU at 3am → anomaly

There are two broad approaches:

| Approach | How it works | Needs labelled anomalies? |
|---|---|---|
| **Supervised** | Train on examples of "normal" AND "anomaly" | Yes — you need lots of labelled examples of both |
| **Unsupervised** | Train only on "normal" data; anything that doesn't fit is flagged | No — just normal data |

We use the **unsupervised** approach because in real factories it's easy to collect hours of normal operation data but very hard (and dangerous) to deliberately cause failures just to get labelled examples.

### 2.3 What is Edge Computing?

**Edge computing** means running computation *close to where the data is produced* — on the factory floor, not in a remote cloud data centre.

**Why run ML on the edge?**

| Advantage | Explanation |
|---|---|
| **Low-latency inference** | Run model inference in milliseconds on-device (no cloud round-trip). End-to-end detection cadence is still window-based (30 samples at 1 Hz by default). |
| **Works offline** | The machine keeps being monitored even if the internet goes down |
| **Data privacy** | Sensitive production data stays on-site |
| **Bandwidth savings** | Send only alerts to the cloud, not thousands of raw readings per second |

**Trade-off**: Edge devices have limited CPU and RAM, so the model must be small and efficient. That's why we use a tiny autoencoder with about 3,482 parameters instead of a large deep-learning model.

### 2.4 What is Docker? (briefly)

**Docker** packages your application and all its dependencies into a **container** — a lightweight, portable box that runs the same way on your laptop, a server, or an edge device.

Think of it like shipping: instead of sending loose items and hoping the recipient has the right shelves, you ship a sealed container with everything inside.

For this project, Docker means:

- You don't need to install Python, TFLite runtime, or paho-mqtt on the edge device
- You just copy the Docker image and run it
- `docker compose up` starts both the MQTT broker and the detector container

---

## 3. How the Algorithm Works

### 3.1 Intuition: The Photocopier Analogy

Imagine a photocopier that has only ever seen pictures of cats.  After copying thousands of cat pictures, it becomes *really good* at reproducing them accurately.

Now feed it a picture of a dog. The copier doesn't know what a dog is — it tries to make it look like a cat. The copy comes out blurry and distorted.

**The key insight**: if the copy is bad, the input was probably not a cat.

An **autoencoder** works the same way:

- **Training**: show it thousands of examples of normal machine behaviour.
- **Inference**: feed it a new reading. If the reconstruction is *close* to the input → **normal**. If the reconstruction is *far* from the input → **anomaly**.

### 3.2 The Math

The autoencoder has two parts: an **encoder** that compresses the data, and a **decoder** that reconstructs it.

**The actual architecture** (not a simple compression):

```
Input Layer    →  Hidden Layer  →  Bottleneck   →  Hidden Layer  →  Output Layer
10 values      →  64 neurons    →  16 neurons   →  64 neurons    →  10 values
   (x)              (expand)        (compress)       (expand)         (x̂)
```

**Why expand first (10→64) before compressing (64→16)?**

The network doesn't work with the raw 10 sensor values directly. Instead:
- **First layer (10→64)**: creates 64 "internal features" — combinations like "high motor current AND high vibration" or "slow feed speed AND high tension." These capture complex relationships between sensors.
- **Bottleneck (64→16)**: forces the network to keep only the 16 most essential patterns that describe "normal" behavior. This is the real compression.
- **Reconstruction (16→64→10)**: rebuilds the original 10 values from those 16 core patterns.

If the network learned well, it can accurately reconstruct normal data from just 16 values. Abnormal data won't fit these 16 patterns, so reconstruction fails → high error → anomaly detected.

---

**Encoder** — compresses through two layers:

1. First layer: expand 10 inputs to 64 features
   $$h_1 = \text{ReLU}(W_1 \cdot x + b_1)$$

2. Second layer: compress 64 features to 16 (the bottleneck)
   $$z = \text{ReLU}(W_2 \cdot h_1 + b_2)$$

Combined: $$z = \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2)$$

**Decoder** — reconstructs the original 10 values from the 16:

3. Third layer: expand bottleneck back to 64
   $$h_3 = \text{ReLU}(W_3 \cdot z + b_3)$$

4. Fourth layer: output 10 reconstructed values
   $$\hat{x} = \sigma(W_4 \cdot h_3 + b_4)$$

Combined: $$\hat{x} = \sigma(W_4 \cdot \text{ReLU}(W_3 \cdot z + b_3) + b_4)$$

---

**Symbol Reference:**

| Symbol | Meaning |
|---|---|
| $x$ | Input: 10 normalised sensor values (one reading from the machine) |
| $h_1$ | Hidden layer: 64 intermediate features (learned combinations of sensors) |
| $z$ | Bottleneck: 16 compressed values (the "essence" of normal behavior) |
| $h_3$ | Hidden layer: 64 features being reconstructed |
| $\hat{x}$ | Output: the reconstruction (same size as $x$, ideally very close to $x$) |
| $W_i, b_i$ | Learned weights and biases (the "knowledge" the network gained during training) |
| $\text{ReLU}(v) = \max(0, v)$ | Activation function: keeps positive values, zeros out negatives (explained below) |
| $\sigma(v) = \frac{1}{1 + e^{-v}}$ | Sigmoid: squashes any value to range $[0, 1]$ to match normalised inputs |

---

**What is ReLU?**

ReLU stands for **Rectified Linear Unit**. It's a simple mathematical function:

$$\text{ReLU}(v) = \max(0, v) = \begin{cases} v & \text{if } v > 0 \\ 0 & \text{if } v \leq 0 \end{cases}$$

**Examples:**
- $\text{ReLU}(-5.0) = 0$
- $\text{ReLU}(-0.1) = 0$
- $\text{ReLU}(0) = 0$
- $\text{ReLU}(2.5) = 2.5$
- $\text{ReLU}(100) = 100$

**Why do we need it?** Without ReLU (or another "activation function"), the neural network would just be a series of matrix multiplications — essentially a fancy linear equation that can only learn straight-line relationships. ReLU adds **non-linearity**, allowing the network to learn curves, thresholds, and complex patterns like "IF motor current > threshold AND vibration > threshold THEN something significant is happening."

**Analogy:** Think of ReLU like a one-way valve. Water (data) can flow forward when pressure is positive, but when pressure is negative, the valve closes (output = 0). This selective behavior helps the network learn which features matter in different situations.

---

**Anomaly score** — Mean Squared Error between input and reconstruction:

$$\text{MSE} = \frac{1}{10} \sum_{i=1}^{10} (x_i - \hat{x}_i)^2$$

**What does this formula do?**

1. For each of the 10 sensors: compute the difference between input $x_i$ and reconstruction $\hat{x}_i$
2. Square each difference: $(x_i - \hat{x}_i)^2$ — this makes all errors positive and penalizes large errors more than small ones
3. Sum all 10 squared differences
4. Divide by 10 to get the average

**Example with normal data:**

| Sensor | Input $x_i$ | Reconstructed $\hat{x}_i$ | Difference | Squared $(x_i - \hat{x}_i)^2$ |
|---|---|---|---|---|
| wire_feed_speed | 0.50 | 0.51 | -0.01 | 0.0001 |
| wire_tension | 0.75 | 0.74 | 0.01 | 0.0001 |
| motor_current_feed | 0.60 | 0.62 | -0.02 | 0.0004 |
| ... (7 more) | ... | ... | ... | 0.0014 |

Sum of squares = 0.0001 + 0.0001 + 0.0004 + 0.0014 = 0.0020
**MSE = 0.0020 / 10 = 0.0002** ← small error = good reconstruction = **normal**

**Example with anomalous data:**

| Sensor | Input $x_i$ | Reconstructed $\hat{x}_i$ | Difference | Squared $(x_i - \hat{x}_i)^2$ |
|---|---|---|---|---|
| motor_current_feed | 0.95 | 0.60 | 0.35 | 0.1225 |
| vibration_rms | 0.90 | 0.30 | 0.60 | 0.3600 |
| temperature_motor | 0.85 | 0.55 | 0.30 | 0.0900 |
| ... (7 more) | ... | ... | ... | 0.0080 |

Sum of squares = 0.1225 + 0.3600 + 0.0900 + 0.0080 = 0.5805
**MSE = 0.5805 / 10 = 0.05805** ← large error = bad reconstruction = **anomaly!**

---

**Threshold** — set during training as:

$$\text{threshold} = \mu_{\text{MSE}} + 3 \cdot \sigma_{\text{MSE}}$$

Where $\mu_{\text{MSE}}$ is the **mean** (average) MSE on training data, and $\sigma_{\text{MSE}}$ is the **standard deviation** (how spread out the MSE values are).

**What is the "three-sigma rule"?**

This rule comes from the **Gaussian distribution** (also called **normal distribution** or **bell curve**), a pattern that appears throughout nature and engineering:

```
    Frequency (how often we see each MSE value)
         ↑
         │               ╱───╲
         │             ╱       ╲          ← Peak: most samples have MSE near the mean
         │           ╱           ╲
         │         ╱               ╲
         │       ╱                   ╲
         │_____╱                       ╲_____
    ─────┴────┬──────┬─────┬───┬─────┬──────┬────→ MSE value
              │      │     │   │     │      │
            μ-3σ   μ-2σ  μ-1σ μ  μ+1σ  μ+2σ  μ+3σ
              │            ↑             │
              │          Mean            │
              │    (average MSE on       │
              │     normal data)         │
              │                          │
              └──────────────────────────┘
                99.7% of normal samples
                    fall in this range


    Understanding the ranges:

    ├────────┤           68% within  μ ± 1σ  (1 standard deviation from mean)
    ├──────────────────┤ 95% within  μ ± 2σ  (2 standard deviations)
    ├────────────────────────────┤ 99.7% within μ ± 3σ  (3 standard deviations)


    Our threshold = μ + 3σ  (right edge of the 99.7% range)
    → Any MSE above this is in the extreme 0.3% tail → ANOMALY!
```

**Key properties of a Gaussian distribution:**

- **68%** of values fall within $\mu \pm 1\sigma$ (within 1 standard deviation)
- **95%** of values fall within $\mu \pm 2\sigma$ (within 2 standard deviations)
- **99.7%** of values fall within $\mu \pm 3\sigma$ (within 3 standard deviations)

**What this means for our detector:**

During training on 10,000 normal samples, suppose we find:
- Mean MSE ($\mu$) = 0.0010
- Standard deviation ($\sigma$) = 0.0008

Then: **threshold** = $0.0010 + 3 \times 0.0008 = 0.0010 + 0.0024 = 0.0034$

**Interpretation:** 99.7% of normal operations have MSE below 0.0034. If a new sample has MSE > 0.0034, it's in the extreme 0.3% tail of the distribution — **statistically very unusual** — so we flag it as an anomaly.

**Why 3 sigma specifically?**

| Threshold | False alarm rate | Trade-off |
|---|---|---|
| $\mu + 2\sigma$ | ~5% (1 in 20 samples) | Too many false alarms |
| $\mu + 3\sigma$ | ~0.3% (1 in 370 samples) | **Good balance** |
| $\mu + 4\sigma$ | ~0.006% (1 in 15,000 samples) | Might miss real problems |

The 3-sigma rule is the engineering sweet spot: sensitive enough to catch real anomalies, but not so sensitive that you get false alarms every few minutes.

### 3.3 Training vs. Inference

| | Training (offline, once) | Inference (real-time, continuous) |
|---|---|---|
| **Input** | CSV of normal data | Live MQTT sensor readings |
| **Goal** | Learn to reconstruct normal data well | Check if new data is normal or not |
| **Output** | Saved model weights + threshold | Anomaly score + alert |
| **Runs** | On your laptop | On the edge device in Docker |

---

## 4. Architecture & Data Flow

```mermaid
flowchart LR
    A["🏭 FTOptix\n(sensor data)"] -->|MQTT publish\nTCP 1883| B["📫 MQTT Broker\n(Mosquitto)"]
    B -->|MQTT subscribe\nTCP 1883| C["mqtt_client.py\n(receive & parse)"]
    C --> D["preprocessing.py\n(buffer → window → normalise)"]
    D --> E["detector.py\n(TFLite autoencoder inference)"]
    E --> F["detector.py\n(score → threshold)"]
    F -->|MQTT publish alert\nanomalous/paperclip/alerts| B
    B -->|MQTT subscribe| G["AnomalyFeedbackManager\n(FTOptix NetLogic)\nparses JSON → sets IsAnomaly"]
    G -->|AnomalyInfoWebAPI\nwrites every 2 s| H["isanomaly.js / .json\n(treeJs_scene/ folder)"]
    H -->|fetch every 2 s\nsame-origin| I["🌐 3-D Digital Twin\n(paper_clip_machine_world.html)\nbeacon · red floods · stack light · operator"]
```

**Step by step:**

1. **FTOptix** reads 10 sensors on the paper-clip machine and publishes a JSON message every second to `ftoptix/paperclip/sensors`.

2. **MQTT Broker** (Mosquitto, running in its own container) receives the message and delivers it to all subscribers.

3. **mqtt_client.py** receives the raw message, parses the JSON, and hands the sensor dictionary to the callback defined in `main.py`.

4. **preprocessing.py** buffers 30 messages (= 30 seconds), averages them into a single vector of 10 values, and normalises each value with the saved scaler.

5. **detector.py** feeds the normalised vector through the TFLite autoencoder and reads the reconstruction.

6. **detector.py** computes the MSE between the original and the reconstruction. If MSE > threshold → anomaly. The result is published back to the MQTT broker on topic `anomaly/paperclip/alerts`.

7. **AnomalyFeedbackManager** (a FTOptix NetLogic) subscribes to `anomaly/paperclip/alerts`, parses the JSON payload, and writes the `IsAnomaly` boolean into the FTOptix information model.

8. **AnomalyInfoWebAPI** (a second FTOptix NetLogic) runs a `PeriodicTask` every 2 seconds. It reads `IsAnomaly` from the model and writes it as a tiny JSON payload into `treeJs_scene/isanomaly.js` on disk.

9. **The 3-D digital twin** (`paper_clip_machine_world.html`, served by FTOptix's embedded WebPresentationEngine on port 8080) polls the state file every 2 seconds with a relative `fetch()`. It probes `./isanomaly.js` first, then `./isanomaly.json`, and keeps whichever answers (the current `AnomalyFeedbackManager.cs` writes `isanomaly.json`). If no answer arrives for 5 seconds — or the page is opened from `file://`, where `fetch()` is unavailable — the scene falls back to its built-in telemetry simulation. When the flag is `true`, the scene raises an alarm that stays readable even at noon: a rotating red beacon with light beams and a red flood light, the flood-light towers switch to red, the stack light glows red, the HMI screen blinks red, a pulsing red vignette frames the view, the whole scene takes on a red alarm grade, and the operator panics.

   **Why a file and not a direct HTTP call?** FTOptix's WebBrowser enforces a strict Content Security Policy (`default-src 'self'`). Any fetch to a different port — even on the same host — is a different origin and is blocked. A relative fetch to a file in the same served folder is always same-origin and requires no special configuration. See the [Decision Log](#6-decision-log) for the full rationale.

---

## 5. Project Walkthrough

### File-by-file guide

| File | Purpose | Key concept |
|---|---|---|
| `config.yaml` | All tuneable parameters in one place | Centralised configuration |
| `src/main.py` | Entry point — wires everything together | Application "conductor" |
| `src/mqtt_client.py` | MQTT subscribe + publish helpers | Pub/sub messaging |
| `src/preprocessing.py` | Buffer samples, window, normalise | Feature engineering |
| `src/model.py` | Autoencoder neural network definition | The ML model itself |
| `src/detector.py` | Compute anomaly score vs. threshold | Decision logic |
| `training/generate_synthetic.py` | Create fake sensor data | Synthetic data generation |
| `training/train.py` | Train the autoencoder offline | Model training |
| `Dockerfile` | Native local recipe with an explicit ARM64 edge build target | Containerisation |
| `docker-compose.yml` | Run broker + detector together (TCP 1883 required; WebSocket 9001 reserved, unused) | Multi-container orchestration |
| `FTOptix/…/ThePaperClipMachineSimulator.cs` | FTOptix NetLogic — generates the 10 simulated sensor tags and publishes them via MQTT every second. Includes an `[ExportMethod]` to inject an anomaly on demand | Sensor simulation |
| `FTOptix/…/AnomalyFeedbackManager.cs` | FTOptix NetLogic — subscribes to the MQTT alert topic, parses the JSON payload, and sets `Model/IsAnomaly` | MQTT → OPC UA bridge |
| `FTOptix/…/AnomalyInfoWebAPI.cs` | FTOptix NetLogic — `PeriodicTask` writes `isanomaly.js` every 2 s from `Model/IsAnomaly` | File-based state export |
| `FTOptix/…/paper_clip_machine_world.html` | 3-D digital twin entry point (Three.js r128): the paper-clip machine outdoors on sand, with paper-clip mountains, a bulldozer and a 24 h day/night cycle in 60 s. Served by FTOptix WebPresentationEngine on port 8080 | Live browser visualisation |
| `FTOptix/…/js/scene/*.js` | The scene code: 12 ES5 classic scripts loaded in order, sharing one `window.PCM` namespace — `core` (renderer, quality tiers, dynamic resolution, URL parameters), `assets` (shared materials, shader hook, procedural textures, geometry helpers), `fx` (glow halos, light cones, particles, ground decals), `sky` (sky dome, sun, moon, stars, clouds, day/night controller), `terrain` (sand, paper-clip mountains, scattered clips), `rockwell` (GuardLogix 5380 safety PLC, Compact 5000 I/O, PowerFlex 525, Stratix, 440G, 450L, OptixEdge and other device faces), `machine`, `characters` (operator, bulldozer driver), `site` (flood-light towers, props, bulldozer), `anomaly` (state polling, alarm effects), `ui` (tour, balloons, hints, title), `main` (build order, frame loop) | Modular, CSP-safe, no build step |
| `FTOptix/…/js/hydrate.js` | Initialises HTML elements from `window.CONTENT` at page load. Extracted from inline `<script>` to satisfy CSP `script-src 'self'` | CSP-compliant hydration |
| `FTOptix/…/js/content.js` | Editable text content for labels, walkthrough text, and stack panel, plus the scene settings `dayNight`, `branding` and `quality` | Content / configuration |
| `FTOptix/…/treeJs_scene/isanomaly.js` | Runtime file written by `AnomalyInfoWebAPI`. Contains `{"isAnomaly":true/false}`. The `.js` extension is required because FTOptix's web server blocks `.json` file downloads | Anomaly state handshake |

### The 10 Most Important Lines of Code

**1. The autoencoder bottleneck** (`src/model.py`):
```python
tf.keras.layers.Dense(bottleneck_size, activation="relu")
```
This is where the magic happens: the network is forced to compress 64 numbers into 16, learning only the most essential patterns.

**2. Buffer emits a window** (`src/preprocessing.py`):
```python
if len(self.samples) == self.window_size:
    window = np.asarray(self.samples, dtype=np.float32)
    self.samples.clear()
    return window
```
Samples accumulate one by one until we have a full window (30 samples), then the buffer resets.

**3. Window averaging** (`src/preprocessing.py`):
```python
return window.mean(axis=0)
```
Collapses 30 samples × 10 tags into one vector of 10 averaged values — smooths noise while keeping the signal.

**4. Normalisation** (`src/preprocessing.py`):
```python
scaled = scaler.transform(feature_vector.reshape(1, -1))
```
The scaler (fitted during training) maps raw values to [0, 1] so no single tag dominates due to its scale.

**5. TFLite inference** (`src/detector.py`):
```python
interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()
reconstructed = interpreter.get_tensor(output_details[0]["index"])[0]
```
Write the input, run the model, read the output — the three-line essence of TFLite inference.

**6. Anomaly score** (`src/detector.py`):
```python
return float(np.mean((original - reconstructed) ** 2))
```
The reconstruction error: low for normal, high for anomalous.

**7. Threshold check** (`src/detector.py`):
```python
return DetectionResult(score > threshold, score, threshold)
```
The decision is still one comparison: is the reconstruction error higher than what we saw during training?

**8. Subscribe on connect** (`src/mqtt_client.py`):
```python
client.subscribe(sensor_topic)
```
Tells the broker: "send me every message on this topic." Done inside `on_connect` so it automatically re-subscribes after reconnection.

**9. Keras training call** (`training/train.py`):
```python
model.fit(x=scaled_data, y=scaled_data, epochs=..., batch_size=...)
```
One call replaces the manual training loop. Keras handles shuffling, batching, forward pass, gradient computation, and weight update on every epoch. The autoencoder's target is its own input (`y=scaled_data`).

**10. Threshold calculation** (`training/train.py`):
```python
threshold = mse_mean + sigma * mse_std
```
The three-sigma rule: anything more than 3 standard deviations above mean training error is an anomaly.

---

## 6. Decision Log

| Decision | Choice | Why | Alternative Considered |
|---|---|---|---|
| Algorithm | Autoencoder | Learns multivariate patterns; intuitive anomaly score (reconstruction error); great for teaching | Isolation Forest — simpler but treats samples independently, less to learn |
| Framework | TensorFlow Lite (TFLite) | Chosen for this project because it integrates with the NXP eIQ/VSI delegate stack on the i.MX 8M Plus target, `tflite-runtime` on the edge device is only a few MB, and it provides a clear split between training (full TF/Keras, dev machine) and inference (TFLite, edge device) | PyTorch — no NPU delegate in this project stack; scikit-learn — can’t easily define custom neural nets |
| Preprocessing | Window averaging (30s, no overlap) | Simple, smooths noise, keeps model input small (10 features) | Raw samples — noisy; sliding window with overlap — adds complexity without much benefit for a tutorial |
| Normalisation | MinMaxScaler to [0, 1] | Matches Sigmoid output; prevents scale imbalance | StandardScaler (z-score) — works too, but [0,1] matches Sigmoid naturally |
| Threshold | mean + 3σ | Classic, easy to explain, well-understood | Fixed threshold — doesn't adapt; percentile — needs more data |
| MQTT QoS | Level 1 (at least once) | Don't lose alerts; QoS 2 adds unnecessary overhead | QoS 0 — might lose alerts; QoS 2 — overkill |
| Docker base | python:3.11-slim | Small image, no compilation issues with pip | Alpine — smaller but C-extension installs often fail |
| Runtime target | ARM64 only | Matches the NXP i.MX 8M Plus deployment target and keeps the runtime stack simple (`tflite-runtime` only) | Multi-arch image logic — more complexity for this tutorial |
| Build platform | `platform: linux/arm64` and `IMAGE_PLATFORM=linux/arm64` in `docker-compose.edge.yml` + `FROM --platform=${IMAGE_PLATFORM}` | Keeps build and runtime architecture explicit even when Podman's delegated Compose builder ignores `build.platforms` | Implicit platform selection — can pull the wrong architecture image |
| Anomaly state → browser | Write `isanomaly.js` to `treeJs_scene/`; browser polls with `fetch('./isanomaly.js')` every 2 s | FTOptix's WebBrowser enforces `default-src 'self'` CSP. Any fetch to a different port (e.g. `http://host:8085/...`) is a cross-origin request and is blocked, even on the same machine. A relative fetch inside the served folder is always same-origin. Worst-case update lag is 4 s (write interval + poll interval), which is acceptable for a visual indicator | `HttpListener` on a dedicated port — blocked by CSP; WebSocket — also cross-origin; MQTT over WebSocket in-browser — requires adding `connect-src ws://...` to the CSP, which is not configurable without access to the FTOptix web server config |
| `.js` extension for the state file | `isanomaly.js` instead of `isanomaly.json` | FTOptix's embedded web server returns 403 for `.json` files — they are on an internal deny list to protect FTOptix project configuration files. Using `.js` avoids the block. The content is still valid JSON; the browser reads it as text and calls `JSON.parse()` explicitly | Rename to `.txt` — also works, but `.js` is more self-documenting in the context of a JS project |
| State file name in the 3-D scene | Probe `./isanomaly.js`, then `./isanomaly.json`; lock onto the one that answers and re-probe after 3 misses | The `.js` decision above and the current `AnomalyFeedbackManager.cs` (which writes `isanomaly.json`) disagree. Auto-detection keeps the scene working with either file, with no configuration | Hard-code one name — the alarm silently stops working when the other file is deployed |
| 3-D scene code structure | 12 ES5 classic scripts in `js/scene/`, one shared `window.PCM` namespace, no build step | The page must run inside FTOptix's WebBrowser under CSP `script-src 'self'` and also from `file://`, where inline scripts and ES modules are blocked. Static parts are merged per material and repeated parts are instanced, so the whole scene renders in about 70 draw calls (shadow pass included) with 41 shared materials | ES modules + bundler — adds a build step and does not load from `file://` |
| 3-D quality tiers | `auto` picks `low` on software renderers (SwiftShader, llvmpipe), `medium` on mobile or ≤ 4-core devices, `high` otherwise; dynamic resolution on top | Panels without GPU acceleration stay interactive (low tier: no MSAA, simpler shadows, fewer instanced clips, 30 fps cap), while PCs get MSAA, soft 2048 px shadows and denser detail. All tiers use the same 7 constant lights, switched by intensity only, so day/night and alarm changes never recompile shaders | One fixed quality — too slow on panels or too plain on PCs |

---

## 7. How to Run

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| [Podman](https://podman.io/) | ≥ 5.0 | Container engine (Docker also works — replace `podman` with `docker` in all commands) |
| Podman Compose / Docker Compose | ≥ 2.0 | Orchestrates the two-container stack |
| Python | 3.11 | For running the integration test and (optionally) training |
| paho-mqtt | ≥ 2.0 | Python package for the test script: `pip install "paho-mqtt>=2.0,<3.0"` |

> **Platform note:**
> The default `docker-compose.yml` is native-platform and can run locally on Windows with Docker Desktop or Podman.
> Use `docker-compose.edge.yml` only when building/running the ARM64 edge image for the NXP target.

### Step 1: Generate synthetic data and train the model

These steps run **on your machine** (not in Docker) because training is a one-time offline task.
Skip this step if the `models/` directory already contains `autoencoder.tflite`, `scaler.pkl`, and `threshold.json`.

1. Create and activate a virtual environment.

```powershell
cd anomaly_detection

# Create virtual environment
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1
```

For Command Prompt (`cmd.exe`):

```bat
venv\Scripts\activate.bat
```

For Linux/macOS:

```bash
source venv/bin/activate
```

2. Install dependencies and run training.

```bash
# requirements.txt covers runtime dependencies; training also needs full TensorFlow
pip install -r requirements.txt
pip install tensorflow

# Generate fake sensor data
python training/generate_synthetic.py

# Train the autoencoder and convert to TFLite
python training/train.py
```

3. (Optional) Deactivate the virtual environment when finished.

```bash
deactivate
```

After training, check the `models/` directory — you should see three files:

```
models/
├── autoencoder.tflite  # compiled TFLite flat-buffer (graph + weights, ~50 KB)
├── scaler.pkl          # fitted MinMaxScaler
└── threshold.json      # anomaly threshold
```

### Step 2: Start the containers

#### 2A) Local run on Windows/Linux/macOS

The default Compose file builds the detector image for your current machine architecture. On Windows with Podman, use either `podman compose` or the Docker-compatible `docker compose` command if your Podman installation provides it.

```powershell
# still inside anomaly_detection/ from Step 1
podman compose up --build

# Equivalent when Podman is exposed through Docker-compatible commands
docker compose up --build
```

#### 2B) Deploy on edge device with Portainer (stack)

> **Important (architecture):** The edge override is ARM64-only: `docker-compose.edge.yml` sets the service platform, build platform, and `IMAGE_PLATFORM` build argument to `linux/arm64`.
> On a Windows amd64 Podman machine, `podman build --platform linux/arm64 ...` can fail with
> `exec container process '/bin/sh': Exec format error` (for example at a `RUN pip install ...` step)
> if ARM emulation is not enabled.
> Use one of these paths:
> - Build on a real ARM64 host (recommended), then export/import with Portainer.
> - Enable ARM emulation in Podman, then build on x86:
>
> ```bash
> podman run --rm --privileged docker.io/tonistiigi/binfmt --install arm64
> podman run --rm --platform linux/arm64 alpine:3.20 uname -m
> # expected output: aarch64
> ```
> - Build a local amd64 image only for development checks:
>
> ```bash
> podman build -t anomaly_detection-detector:amd64-local .
> ```
>
> The amd64-local image is for local validation only; deploy `anomaly_detection-detector:edge` on the ARM64 device.

1. Build and export ARM64 images on your development machine:

```bash
# from anomaly_detection/
podman compose -f docker-compose.yml -f docker-compose.edge.yml build
podman pull --platform linux/arm64 eclipse-mosquitto:2
podman save -o edge-images.tar anomaly_detection-detector:edge eclipse-mosquitto:2
```

2. Copy these files to the edge device:
- `edge-images.tar`

3. In Portainer on the edge device:
- Go to **Images** → **Import** and upload `edge-images.tar`
- If Portainer shows the detector image as `localhost/anomaly_detection-detector:edge`, this is expected for archives created by Podman. In that case, set the detector image in your stack YAML to `localhost/anomaly_detection-detector:edge` (instead of `anomaly_detection-detector:edge`).
- Go to **Stacks** → **Add stack**
- Paste this stack file and deploy:

```yaml
services:
  mqtt-broker:
    image: eclipse-mosquitto:2
    ports:
      - "1883:1883"   # MQTT TCP — FTOptix + detector container
      - "9001:9001"   # Optional: reserved for future browser MQTT-over-WebSocket sync
    environment:
      - MOSQUITTO_LISTENER=1883 0.0.0.0
      - MOSQUITTO_ALLOW_ANONYMOUS=true
    command:
      - /bin/sh
      - -c
      - |
        cat > /tmp/mosquitto.conf <<EOF
        listener $${MOSQUITTO_LISTENER}
        allow_anonymous $${MOSQUITTO_ALLOW_ANONYMOUS}
        listener 9001 0.0.0.0
        protocol websockets
        EOF
        exec mosquitto -c /tmp/mosquitto.conf
    healthcheck:
      test: ["CMD-SHELL", "mosquitto_sub -h 127.0.0.1 -p 1883 -t '$$SYS/broker/version' -C 1 -W 3 >/dev/null 2>&1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    restart: unless-stopped

  detector:
    image: localhost/anomaly_detection-detector:edge
    pull_policy: never
    platform: linux/arm64
    environment:
      - PYTHONUNBUFFERED=1
      - TF_CPP_MIN_LOG_LEVEL=3
      - TF_ENABLE_ONEDNN_OPTS=0
    restart: unless-stopped
    depends_on:
      mqtt-broker:
        condition: service_healthy
```

5. Open the 3-D digital twin in a browser on the same network:

```
file:///.../FTOptix/paper_clip_machine/ProjectFiles/treeJs_scene/paper_clip_machine_world.html
```

Or serve it from any static HTTP server (e.g. VS Code Live Server, Nginx, or FTOptix's own embedded web server). The page currently runs as a standalone 3-D visualisation while MQTT browser sync is disabled. Opened from `file://`, the page cannot `fetch()` the state file and runs on its built-in simulation; served over HTTP, it follows `isanomaly.js` / `isanomaly.json`.

Optional URL parameters for demos and debugging (for example `paper_clip_machine_world.html?time=19.2&anomaly=1`):

| Parameter | Values | Effect |
|---|---|---|
| `quality` | `auto` (default), `high`, `medium`, `low` | Render quality tier; overrides `quality.tier` in `content.js` |
| `time` | `0`–`24` | Freeze the day/night clock at this hour (e.g. `12` noon, `18.2` sunset, `0` midnight) |
| `speed` | number, default `1` | Day/night speed multiplier; `1` = 24 h in 60 s (`dayNight.cycleSeconds`) |
| `anomaly` | `1` / `0` | Force the alarm on / off, ignoring the state file |
| `stats` | `1` | Show an overlay with tier, FPS, frame time, draw calls, triangles, pixel ratio and GPU name |
| `dynres` | `0` | Disable dynamic resolution (useful for benchmarks) |

4. Validate logs in Portainer:
- `mqtt-broker` shows Mosquitto startup
- `detector` shows `Model autoencoder.tflite (float32) on CPU ...` and `Anomaly detector running`

To run the detector on the NPU instead, see [Part 3, section 33](#33-deploying-to-the-device).

### Step 3: Run the integration test (automated)

The simplest way to verify the running system is the Python integration test. It publishes 30 normal samples, waits for a detection result, then publishes 30 anomalous samples and verifies an anomaly is detected.

```bash
# From the anomaly_detection/ directory (stack must be running)
python tests/test_integration.py
```

Expected output:

```
============================================================
  Anomaly Detection — Integration Test
  Broker: localhost:1883
  Sensor topic : ftoptix/paperclip/sensors
  Alert  topic : anomaly/paperclip/alerts
  Window size  : 30 messages
============================================================

[TEST] Normal window
  [NORMAL] Publishing 30 normal sensor samples ...
  [NORMAL] PASS — NORMAL ✓  score=0.000215  threshold=0.007683

[TEST] Anomaly window
  [ANOMALY] Publishing 30 anomalous sensor samples ...
  [ANOMALY] PASS — ANOMALY DETECTED ✓  score=166.113953  threshold=0.007683

============================================================
  RESULT: ALL TESTS PASSED ✓
============================================================
```

> **Required:** `pip install "paho-mqtt>=2.0,<3.0"` in your local Python environment.

---

### Step 4: Simulate sensor data manually (optional)

**Understanding the sensor tags:**

Each JSON message contains 10 sensor readings from the paper-clip machine. Here's what each one measures:

| Tag | What it measures | Unit | What's happening physically |
|---|---|---|---|
| **wire_feed_speed** | How fast the steel wire is being pulled into the machine | m/min | A motor-driven roller feeds wire from a spool. Too fast → bad bends; too slow → low throughput |
| **wire_tension** | How taut the wire is as it enters the machine | N (Newtons) | The wire must be under consistent tension to bend cleanly. Too loose → sloppy clips; too tight → wire snaps |
| **motor_current_feed** | Electrical current drawn by the **feed motor** (the one pulling wire) | A (Amps) | Higher current = motor working harder. A spike could mean a jam or worn bearings |
| **motor_current_bend** | Electrical current drawn by the **bending motor** (the one shaping the clip) | A (Amps) | Same idea — if bending becomes harder (e.g. wrong wire gauge, worn tooling), current rises |
| **bend_angle** | The angle at which the wire is bent to form the clip shape | ° (degrees) | A paper clip has specific bends (~180°). Deviations mean the tooling is misaligned or worn |
| **cycle_time** | Time to produce one complete paper clip | ms | A healthy machine has a consistent rhythm. Slower cycles could indicate a mechanical issue |
| **vibration_rms** | Overall vibration intensity (RMS = root mean square, a standard vibration measure) | mm/s | All machines vibrate, but excessive vibration signals loose parts, imbalance, or bearing wear |
| **cut_force** | The force applied by the blade that cuts the wire after bending | N (Newtons) | Consistent cutting force = sharp blade. Rising force = blade is dulling or wire is harder than expected |
| **temperature_motor** | Temperature of the main motor | °C | Motors heat up during operation. Overheating means overload, poor cooling, or failing bearings |
| **reject_count_per_min** | Number of defective clips detected per minute | count | Quality sensors (e.g. vision camera) count bad clips. A sudden rise means something has gone wrong |

Together, these 10 tags cover the key failure modes: **mechanical wear** (vibration, motor current, cut force), **process drift** (bend angle, cycle time, wire tension), **thermal issues** (motor temperature), and **quality impact** (reject count).

In a **separate terminal**, publish a test message:

```bash
# Install mosquitto-clients (or use any MQTT client)

# Publish a NORMAL reading:
mosquitto_pub -h localhost -t "ftoptix/paperclip/sensors" -m '{
  "timestamp": "2026-02-17T10:30:01.000Z",
  "wire_feed_speed": 10.2,
  "wire_tension": 22.5,
  "motor_current_feed": 2.1,
  "motor_current_bend": 3.3,
  "bend_angle": 180.1,
  "cycle_time": 198,
  "vibration_rms": 1.8,
  "cut_force": 55.0,
  "temperature_motor": 48.3,
  "reject_count_per_min": 1
}'
```

Publish 30 messages (one window) to trigger a detection. After the 30th message you'll see either:

```
detector  | ✅ Normal  score=0.000215  threshold=0.007683
```

or (if you send abnormal values):

```
detector  | 🚨 ANOMALY DETECTED  score=166.113953  threshold=0.007683
```

### Simulating an anomaly

Publish a message with extreme values (e.g., motor current at 15 A instead of the normal 2–4 A):

```bash
mosquitto_pub -h localhost -t "ftoptix/paperclip/sensors" -m '{
  "timestamp": "2026-02-17T10:31:01.000Z",
  "wire_feed_speed": 10.0,
  "wire_tension": 22.0,
  "motor_current_feed": 15.0,
  "motor_current_bend": 12.0,
  "bend_angle": 170.0,
  "cycle_time": 350,
  "vibration_rms": 9.0,
  "cut_force": 120.0,
  "temperature_motor": 90.0,
  "reject_count_per_min": 8
}'
```

---

## 8. Experimenting & Next Steps

### Parameters to tweak in `config.yaml`

| Parameter | What to try | Effect |
|---|---|---|
| `window_size` | 10, 30, 60 | Smaller = faster detection but noisier; larger = smoother but slower |
| `bottleneck_size` | 4, 8, 16, 32 | Smaller = more compression = less sensitive; larger = less compression = more sensitive |
| `threshold_sigma` | 2.0, 3.0, 4.0 | Lower = more alerts (more sensitive); higher = fewer alerts (less sensitive) |
| `epochs` | 50, 100, 200 | More epochs = better fit but risk of overfitting; watch the training loss plateau |
| `learning_rate` | 0.0001, 0.001, 0.01 | Too low = slow training; too high = unstable training |

---

# Part 2 — Gesture Control

> Sections 9 to 25: a camera, a ready-made pose model and a few geometric rules let an operator control the HMI with four hand gestures. The Python service is built and tested on a laptop; the FactoryTalk Optix side and the NPU are planned.

## 9. Gesture Control Overview

### 9.1 What it does

An operator stands in front of a camera and controls the HMI without touching it. The service recognizes four gestures:

| Gesture (name in the messages) | What the operator does | What the HMI will do |
|---|---|---|
| `open_palm` | Raises one hand to shoulder or face height, forearm up, and holds it still for 1 second, like a "stop" sign | Open a dialog |
| `palm_up` | Pushes one hand quickly upwards until it is above the head | Show a message |
| `swipe_left` | Moves one hand quickly to their own left, between chest and head height | Go to the previous page |
| `swipe_right` | Moves one hand quickly to their own right, between chest and head height | Go to the next page |

"Left" and "right" are seen from the operator's side, as in a mirror ([13.3](#133-body-coordinates-in-shoulder-widths)). Either hand can make any gesture.

Two honest limits up front:

- **No fingers.** The pose model finds wrists and elbows but no fingers, so it cannot tell an open hand from a fist. In this project, "open palm" means "a raised, still hand with the forearm pointing up" ([10.4](#104-pose-estimation-and-keypoints)).
- **The HMI side is not built yet.** The actions in the last column are designed ([section 16](#16-factorytalk-optix-integration-planned)) but not implemented. Today the service publishes MQTT messages, which you can watch with an MQTT client ([23.7](#237-watch-the-messages)).

### 9.2 What is built and what is not

Legend: ✅ done and verified · ⚠️ partly verified · 📝 designed or written, not run · ❌ not started

| Part | Status | Evidence |
|---|---|---|
| Gesture service: camera → MoveNet → rules → MQTT | ✅ | 60 unit tests pass, natively and inside the container |
| Laptop run with Podman and the built-in webcam | ✅ | 19.5 frames per second; preparing the image and running MoveNet take about 3.2 ms per frame on the laptop CPU ([19.1](#191-measured-on-the-laptop)) |
| Health reporting and camera reconnect | ✅ | Camera stream stopped → `camera_lost` after about 2 s → `running` again when it returns; service killed → `offline` after about 1 s ([17.5](#175-who-notices-what)) |
| Gestures made by a real person in front of the camera | ⚠️ | The rules are tested only with synthetic (computer-generated) poses. The thresholds are reasoned design values, not yet tuned on recordings |
| FactoryTalk Optix side: subscriptions, NetLogic, pages, dialog | 📝 | Design only ([section 16](#16-factorytalk-optix-integration-planned)) |
| NPU acceleration on the i.MX 8M Plus | 📝 | Code path written. Its fallback to the CPU is tested on the laptop; it has never run on the device |
| ARM64 image, edge compose file, Ethernet IP camera | ❌ | Roadmap ([25.3](#253-roadmap)) |

### 9.3 The big idea in one paragraph

About 20 times per second, the service takes the newest picture from the camera, shrinks it to 192 × 192 pixels and gives it to **MoveNet**, a small neural network published by Google. MoveNet answers with 17 body points (nose, eyes, ears, shoulders, elbows, wrists, hips, knees and ankles), each one a position plus a confidence score: 51 numbers per picture. From then on no AI is involved. Plain Python rules look at how the wrists moved, relative to the shoulders, during the last second or so, and decide whether that movement was one of the four gestures. When a rule fires, the service publishes one small JSON message over MQTT and then stops listening until the gesture is over, so one gesture never triggers twice. Pictures are never saved to disk; only gesture events and a health status leave the service.

**In one sentence:** a pre-trained model turns pictures into body points, and simple, testable rules turn body-point movements into commands.

### 9.4 How Part 2 differs from Part 1

| | Part 1: anomaly detection | Part 2: gesture control |
|---|---|---|
| Input | 10 sensor values once per second, over MQTT | Camera pictures, up to 20 per second |
| Model | An autoencoder we train ourselves (about 3,482 parameters) | MoveNet, trained by Google (a 2.9 MB file); we never train it |
| What the model outputs | How well it can rebuild its input (the reconstruction error) | 17 body points, each with a score |
| Decision | One threshold (three-sigma rule) | Geometric rules over time, plus repeat protection |
| MQTT output | `anomaly/paperclip/alerts` | `gesture/paperclip/events` and `gesture/paperclip/status` |
| FactoryTalk Optix side | `AnomalyFeedbackManager` NetLogic sets `IsAnomaly` (built) | `GestureEventManager` NetLogic (planned) |
| Where the model runs | ARM CPU (NPU not enabled yet) | Laptop CPU today; the i.MX 8M Plus NPU is planned |

## 10. Machine Learning Basics for Gesture Control

This section explains every idea that Part 2 needs, starting from zero. If you read Part 1, sections 10.1 and 10.2 will feel familiar.

### 10.1 What is a neural network

A **neural network** is a function: numbers go in, numbers come out. Inside, it is a long chain of very simple arithmetic (multiply, add, and a simple bend such as ReLU, which turns negative numbers into 0) organized in **layers**. The numbers it multiplies by are called **weights** or **parameters**. Part 1 shows the arithmetic of a single neuron step by step ([3.2 The Math](#32-the-math)).

Think of a huge **mixing desk** with thousands of knobs. The input signal goes in, every knob changes how it is mixed, and something comes out. **Training** means turning the knobs a tiny bit at a time until, for thousands of examples, the output matches the right answer. After training the knobs are frozen and saved in the model file.

For pictures, the main building block is the **convolution**. A small window, say 3 × 3 pixels, slides over the picture. At every position it multiplies the 9 pixel values by 9 weights and adds the results. Where the picture contains the pattern those weights "look for" (an edge, a corner), the sum is large. Stacking many such layers lets the network combine small patterns into bigger ones, until a late layer can say "an elbow is probably here".

We counted the operations inside our MoveNet file: 157 in total, of which 50 are convolutions and 24 are **depthwise** convolutions, a cheaper variant that filters each channel separately. NXP's model page describes MoveNet as "a lightweight human pose estimation model designed to run in real time on mobile devices", with a MobileNetV2 network as its image-reading part ([eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo/blob/main/tasks/vision/pose-estimation/movenet/README.md)).

### 10.2 Training vs inference, and why we never train

- **Training** shows the network many labelled examples and adjusts the weights. It needs a lot of data and computing power, and it is done once.
- **Inference** uses the finished, frozen network on new input. It is cheap enough to run 20 times per second.

MoveNet was trained by Google on two datasets: [COCO](https://cocodataset.org/), a public collection of photos in which people marked the body points by hand, and an internal Google dataset called Active ([MoveNet blog](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)). We download the finished 2.9 MB file and **only run inference**, which is why `gesture_recognition/` has no training folder. Part 1 is different: there we train the autoencoder ourselves ([3.3](#33-training-vs-inference)). Google publishes the model under the Apache 2.0 license ([Kaggle model page](https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8)).

### 10.3 What a computer sees in an image

- A picture is a grid of **pixels**. Each pixel is 3 numbers (red, green, blue), each from 0 to 255.
- The webcam bridge asks the webcam for pictures of 640 × 480 pixels: 640 × 480 × 3 = **921,600 numbers** per picture.
- Positions are counted from the **top-left** corner: x grows to the right and **y grows downwards**. A hand moving up therefore gets a *smaller* y. Our rules flip y so that up is positive ([13.3](#133-body-coordinates-in-shoulder-widths)).
- OpenCV, the library that reads the camera, stores the colours in the order blue, green, red (BGR) ([OpenCV docs](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html)). MoveNet expects red, green, blue (RGB), so the service swaps them ([12.3](#123-preparing-the-image-for-the-model)).
- MoveNet reports positions as **fractions** of the picture, from 0 to 1, not in pixels: x = 0.5 is the middle, y = 0.25 is a quarter of the way down.

### 10.4 Pose estimation and keypoints

**Pose estimation** means finding where a person's body parts are in a picture. The points it finds are called **keypoints**. MoveNet finds 17 of them, always in the same order ([TensorFlow MoveNet tutorial](https://www.tensorflow.org/hub/tutorials/movenet)):

| Index | Keypoint | Used by our rules? |
|---|---|---|
| 0 | nose | no |
| 1, 2 | left eye, right eye | no |
| 3, 4 | left ear, right ear | no |
| 5, 6 | left shoulder, right shoulder | **yes**: who is the operator, and the ruler for all hand positions |
| 7, 8 | left elbow, right elbow | **yes**: the "forearm up" check of `open_palm` |
| 9, 10 | left wrist, right wrist | **yes**: the hand positions that make the gestures |
| 11, 12 | left hip, right hip | no |
| 13, 14 | left knee, right knee | no |
| 15, 16 | left ankle, right ankle | no |

Each keypoint is 3 numbers: **y, x and a score** from 0 to 1 that says how sure the model is (higher = more sure). For example, `(0.25, 0.80, 0.15)` means "a quarter of the way down, 80 % of the way to the right, but only 15 % sure". Our rules ignore every keypoint with a score below 0.3 (`min_keypoint_score`).

"Left" and "right" are the **person's own** left and right, not the sides of the picture. A person facing the camera has their right wrist on the left side of the picture.

There are **no finger keypoints**: a wrist is one point. That is why the service cannot tell an open hand from a fist.

### 10.5 MoveNet

Google published MoveNet in May 2021 ([MoveNet blog](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)). The single-person version comes in two sizes:

| Variant | Input size | Google's description |
|---|---|---|
| **Lightning** (we use this) | 192 × 192 | "intended for latency-critical applications" |
| Thunder | 256 × 256 | for applications that need higher accuracy |

A separate MultiPose version "detects up to 6 poses" ([TensorFlow.js MoveNet page](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/movenet/README.md)). We use **SinglePose Lightning, int8, version 4** from TensorFlow Hub, which is also where NXP's [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo/blob/main/tasks/vision/pose-estimation/movenet/README.md) downloads it. The zoo page says this int8 model "has been tested on i.MX 8MP", but it publishes no speed figure. `tools/fetch_model.py` downloads it and checks its SHA-256 fingerprint, so every machine runs a byte-identical file (2,894,840 bytes).

**How it works inside (simplified from the blog):**

1. A **feature extractor** (MobileNetV2 with a "feature pyramid network") reads the picture and builds a summary of what is where, at several zoom levels.
2. Four small **prediction heads** read that summary. They produce a **heat map** of person centres (a grid of scores where high means "probably here"), a first guess of all 17 keypoints for each person, a heat map for each of the 17 keypoint types, and small **offsets** that place each point more precisely than one grid cell.
3. SinglePose picks the highest-scoring person centre, giving more weight to people near the middle of the frame. Starting from that person's first guess, it takes the strongest nearby peak in each keypoint heat map and adds the offset. In our file the heat maps are 48 × 48 cells, so one cell covers 4 × 4 input pixels (we read this from the model's tensors).

**What our service does not use:** the browser (TensorFlow.js) version of MoveNet adds "intelligent cropping based on detections from the previous frame", which zooms in on the person found there, and smoothing over time. Our service has neither: each frame is processed on its own, and the gesture rules deal with jitter instead.

**Input and output, as reported by the interpreter:**

| | Shape | Type | Meaning |
|---|---|---|---|
| Input | `[1, 192, 192, 3]` | uint8 (0 to 255) | 1 picture, 192 rows, 192 columns, 3 colours (RGB) |
| Output | `[1, 1, 17, 3]` | float32 | 1 picture, 1 person, 17 keypoints, (y, x, score) each from 0 to 1 |

### 10.6 Quantization

Networks are normally trained with 32-bit decimal numbers ("floats"). **Quantization** stores the weights and the intermediate values as 8-bit integers, from −128 to 127, instead. Each number then takes a quarter of the space, and integer arithmetic is cheaper. The "int8" in our model's name means exactly this.

How can a small integer stand for a decimal like 0.123? Every quantized tensor (a block of numbers) carries two settings, a **scale** and a **zero point** ([LiteRT quantization spec](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec)):

```
real value = (stored integer − zero_point) × scale
```

Think of a **ruler with 256 marks**. The scale is the distance between two marks; the zero point says which mark means 0.

**Worked example**, with scale 0.02 and zero point 0:

- The stored integer 25 means (25 − 0) × 0.02 = **0.5**.
- The whole ruler reaches from −128 × 0.02 = −2.56 to 127 × 0.02 = 2.54.
- To store 0.123: 0.123 ÷ 0.02 = 6.15, rounded to the nearest mark = 6, which reads back as **0.12**. The error is at most half a mark, here 0.01.

**Per-tensor or per-channel.** A tensor can use one ruler for all its numbers (**per-tensor**) or a separate ruler for each output channel, one per filter (**per-channel**), which fits each filter better. In our file, 146 tensors use per-channel scales and 143 use a single scale (counted with the interpreter). This matters on the NPU: a reference project for the i.MX 8M Plus found that per-tensor quantization, which it calls "the NPU-optimal mode", collapses MoveNet's keypoint confidence to about 0, so they keep per-channel ([Avocado imx8mp-npu-pose](https://github.com/avocado-linux/references/blob/main/imx8mp-npu-pose/README.md)). Our plan is the same: keep the original file and measure it first ([19.4](#194-how-the-npu-will-be-used)).

**What this means for our code:** the model's input is plain uint8 pixel values (the interpreter reports no quantization settings for it), and the model's first operation converts them itself. The output is already float32. So the service passes the RGB picture in directly and reads ordinary decimals out, with no quantization arithmetic in Python.

### 10.7 CPU, NPU, runtime and delegates

- The **CPU** is the general-purpose processor. The i.MX 8M Plus has 4 Arm Cortex-A53 cores at up to 1.8 GHz, plus a separate Cortex-M7 core ([board documentation](https://github.com/zephyrproject-rtos/zephyr/blob/main/boards/variscite/imx8mp_var_som/doc/index.rst)). On the edge device the same CPU also runs FactoryTalk Optix, so every millisecond we save is left for the HMI.
- The **NPU** (Neural Processing Unit) is a block of the chip built to do huge numbers of multiply-and-add operations in parallel, which is what convolutions are made of. The i.MX 8M Plus NPU is rated at 2.3 TOPS (trillion operations per second, a best-case peak figure) ([board documentation](https://github.com/peridio/docs/blob/main/src/docs-hardware/variscite/dart-mx8m-plus/index.mdx)).
- Analogy: the CPU is a **Swiss-army knife** that can do anything; the NPU is a **power drill**, much faster at one job and useless for the rest.
- The **runtime** is the program that loads a model file and executes it. We use **LiteRT**, Google's new name for TensorFlow Lite ([announcement](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/)), installed as the Python package `ai-edge-litert`.
- A **delegate** is a plug-in that lets the runtime hand parts of the model to an accelerator. Think of a manager (the runtime) who gives the specialist (the delegate, driving the NPU) every job the specialist can do, and does the rest personally on the CPU. Operations the delegate cannot run stay on the CPU ([LiteRT delegates](https://ai.google.dev/edge/litert/performance/delegates)). NXP's delegate for this NPU is the **VX delegate**, a library at `/usr/lib/libvx_delegate.so` on NXP's Linux images ([NXP demo](https://github.com/nxp-imx/efficientdet-imx/blob/lf-5.15.5_1.0.0/DEMO_README.md)).
- **Our code** tries the VX delegate only when `use_npu` is on, runs one test inference with it, and on any error logs a warning and uses the CPU ([12.4](#124-running-movenet-and-reading-the-result)). The MQTT status reports which one is in use (`"accelerator": "npu"` or `"cpu"`).

### 10.8 Video basics

- A video is a sequence of pictures called **frames**; **fps** means frames per second. The webcam bridge asks the webcam for 30 fps, and the service analyses at most 20 fps (`target_fps`).
- Cameras **compress** video. Two common formats:
  - **MJPEG** (Motion JPEG): every frame is a separate JPEG photo. Simple, and every frame stands alone, but it needs more bandwidth. Our webcam bridge serves MJPEG over HTTP as one never-ending response, one JPEG after the other.
  - **H.264**: stores mostly the *differences* between frames. It needs much less bandwidth, but each frame depends on earlier ones, so the decoder is more complex.
  - **RTSP** is the protocol IP cameras use to offer their streams (`rtsp://...` addresses); the video inside is usually H.264 or H.265.
- **Decoding** turns the compressed data back into pixels, and it costs CPU for *every* frame the camera sends, including the frames the service then skips. The i.MX 8M Plus has a hardware video decoder (VPU) for H.265, H.264, VP9 and VP8 up to 1080p at 60 fps ([board documentation](https://github.com/peridio/docs/blob/main/src/docs-hardware/variscite/dart-mx8m-plus/index.mdx)). The service does not ask OpenCV for hardware decoding, so it does not use the VPU ([19.5](#195-bottlenecks)).
- **OpenCV** reads all of these with one function, `cv2.VideoCapture`: a webcam number, an HTTP MJPEG stream, an RTSP stream or a video file.

## 11. Gesture Architecture

This section shows where each piece runs: first the planned production setup, then the laptop setup that runs today, then the inside of the service.

### 11.1 Production on the i.MX 8M Plus

> 📝 **Planned.** This is the target design. None of it has run on the device yet ([9.2](#92-what-is-built-and-what-is-not)).

```mermaid
flowchart LR
  CAM["Ethernet IP camera<br/>RTSP or MJPEG"]
  subgraph EDGE["i.MX 8M Plus edge device"]
    subgraph SVC["Container: gesture service"]
      GRAB["Camera thread<br/>keeps the newest frame"]
      POSE["MoveNet<br/>17 keypoints"]
      RULES["Gesture rules"]
      PUB["MQTT publisher"]
    end
    NPU[("NPU<br/>through the VX delegate")]
    BROKER["Mosquitto broker<br/>shared with Part 1"]
    subgraph OPTIX["FactoryTalk Optix runtime"]
      MQ["MQTT client"]
      NL["GestureEventManager<br/>NetLogic"]
      UI["Pages, dialog, message"]
    end
  end
  SCREEN["Browser or panel"]
  CAM -->|frames| GRAB
  GRAB --> POSE
  POSE --> RULES
  RULES --> PUB
  POSE <-.->|inference| NPU
  PUB -->|events and status| BROKER
  BROKER --> MQ
  MQ --> NL
  NL --> UI
  UI --> SCREEN
```

Reading the diagram from left to right:

1. An **Ethernet IP camera** sends its video stream (RTSP, or MJPEG over HTTP) to the edge device.
2. The **gesture service** runs in a container. It keeps only the newest frame, runs MoveNet on it (on the NPU when possible, otherwise on the CPU), applies the gesture rules and publishes MQTT messages.
3. The **Mosquitto broker** is the one Part 1 already uses, so no second broker is needed.
4. **FactoryTalk Optix** receives the messages with its built-in MQTT client. A NetLogic called `GestureEventManager` checks each message and performs the action. The result appears in every browser or panel that shows the HMI.

Planned device settings, each explained in the linked section:

| Setting | Planned value | Why |
|---|---|---|
| Container network | `network_mode: host`: the container shares the device's network | The broker is reachable at `127.0.0.1:1883`, the same address FactoryTalk Optix uses, and the camera's video does not pass through the container network's address translation (NAT), which can block RTSP video sent over UDP. The service opens no listening ports, so sharing the network exposes nothing new ([22](#22-gesture-decision-log)) |
| `USE_NPU` | `true` | Use the NPU. The code falls back to the CPU by itself if the NPU fails ([12.4](#124-running-movenet-and-reading-the-result)) |
| Camera stream | A low-resolution substream, for example 640 × 360 at about 20 fps | The CPU must decode every frame the camera sends ([19.5](#195-bottlenecks)) |
| Camera address | `CAMERA_URL`, set in the Portainer stack | Camera passwords never go into Git |

### 11.2 Development on a Windows laptop

> ✅ **Runs today.** All measurements in this guide come from this setup: a laptop with an Intel Core i7-12800H, Windows and Podman 5.8.1.

```mermaid
flowchart LR
  subgraph WIN["Windows laptop"]
    CAM["Built-in webcam"]
    BRIDGE["webcam_bridge.py<br/>runs in Windows Python<br/>MJPEG over HTTP, port 8090"]
    OPTIX["FactoryTalk Optix"]
    subgraph VM["Podman machine: a WSL2 Linux virtual machine"]
      SVC["Container: gesture service<br/>CPU only"]
      BROKER["Container: Mosquitto broker<br/>port published"]
    end
  end
  CAM --> BRIDGE
  BRIDGE -->|WSL adapter address| SVC
  SVC -->|host.containers.internal| BROKER
  OPTIX -->|localhost| BROKER
```

1. **`webcam_bridge.py`** runs on Windows, not in a container. It reads the built-in webcam and serves the pictures as an MJPEG stream on port 8090.
2. The **gesture service** container runs inside the **Podman machine**, a small Linux virtual machine that Podman creates with WSL2 (Windows Subsystem for Linux). It reads the stream at the **WSL adapter address**, the Windows side of the virtual network between Windows and WSL2.
3. The **broker** container also runs in the Podman machine. Its port is published, so FactoryTalk Optix on Windows reaches it at `localhost`, and the gesture container reaches it at `host.containers.internal`.

Why the extra bridge program? The WSL2 virtual machine cannot see the laptop's USB webcam unless extra tools pass the device through. The bridge turns the webcam into the same kind of input an Ethernet camera gives, so the container code is identical on the laptop and on the device; only the camera address changes. The details, and the four networking rules that make this work, are in [18.3](#183-why-the-webcam-needs-a-bridge) and [18.4](#184-networking-between-windows-podman-and-containers).

You can also run the service directly with Windows Python, reading the webcam itself instead of the bridge ([23.6](#236-build-and-start-the-container)).

### 11.3 Inside the service, four threads

A **thread** is an independent line of work inside one program. Threads run at the same time and share the program's memory. The service uses four:

| Thread | Code | Job | Why it is separate |
|---|---|---|---|
| Camera | `camera.py`, `Camera._run` | Reads frames non-stop, keeps only the newest, reconnects when the camera is lost | Reading must never wait for the model, or old frames pile up ([12.1](#121-keeping-only-the-newest-frame)) |
| Main | `main.py`, `main()` | Takes the newest frame, runs MoveNet and the rules, publishes events, writes the status and the heartbeat | All gesture logic runs in this one thread, so the rules need no locks |
| MQTT network | started by paho's `loop_start()` | Sends and receives MQTT packets and reconnects to the broker | Publishing never blocks the main loop |
| Watchdog | `main.py`, `Watchdog` | Ends the process if the main loop stops making progress | A hung loop cannot notice that it hangs; a separate thread can ([17.4](#174-internal-watchdog)) |

### 11.4 One frame's journey

1. **Camera thread** ([12.1](#121-keeping-only-the-newest-frame)): OpenCV reads the next JPEG from the stream and decodes it into an array of numbers (640 × 480 × 3 with the bridge's default settings). The array goes into the latest-frame slot together with a frame number and a timestamp, replacing the previous frame.
2. **Main thread**: takes the newest frame it has not processed yet, waiting at most 0.5 s.
3. **Preprocess** ([12.3](#123-preparing-the-image-for-the-model)): cuts out the middle 480 × 480 square, shrinks it to 192 × 192 and swaps BGR to RGB.
4. **MoveNet** ([12.4](#124-running-movenet-and-reading-the-result)): returns 17 keypoints. On the laptop CPU this takes about 3 ms.
5. **Gesture rules** ([13](#13-from-keypoints-to-gestures) and [14](#14-one-gesture-one-event)): presence gate, body coordinates, wrist tracks, detectors, latch. This takes tens of microseconds.
6. **Publish** ([15](#15-mqtt-messages-for-gestures)): if a gesture fired, one JSON message goes to `gesture/paperclip/events`.
7. **Rate cap** ([12.2](#122-the-rate-cap)): if the work took less than 50 ms, the loop waits for the rest, so it runs at most 20 times per second.
8. **Every 2 seconds**, the loop also publishes a status message and rewrites the heartbeat file ([17.3](#173-heartbeat-file-and-container-healthcheck)).

Two statistics appear throughout Part 2. **p50** (the median) is the time that half of the frames beat. **p95** is the time that 95 % of the frames beat, so it shows the slow cases. On the laptop, from "frame received" to "decision made" took 33 to 35 ms at p95. Most of that is the frame waiting while the loop sleeps for the rate cap, not computing ([12.2](#122-the-rate-cap)).

## 12. From Camera Frame to Keypoints

### 12.1 Keeping only the newest frame

**Thought experiment.** The webcam delivers 30 frames per second, and the service analyses 20. Suppose the frames waited in a queue, first in, first out, like letters in a mailbox. The queue grows by 10 frames every second. After 10 seconds, 100 frames are waiting, which is 100 ÷ 20 = 5 seconds of work. A swipe would change the page 5 seconds late, and the delay would keep growing.

The fix is a **scoreboard instead of a mailbox**: a scoreboard shows only the current score. The camera thread reads every frame as soon as it arrives and keeps only the newest one in a single slot. The main thread always takes whatever is newest, so the delay can never build up. Reading continuously also stops OpenCV's own buffers from filling with old frames.

From `camera.py`, the camera thread storing a frame:

```python
            with self._cond:
                self._seq += 1
                self._latest = (frame, time.perf_counter(), self._seq)
                self.state = RUNNING
                self._cond.notify_all()
```

and the main thread taking it:

```python
    def latest(self, after_seq, timeout):
        """Newest (frame, t, seq) with seq > after_seq, or None after timeout s."""
        with self._cond:
            if self._cond.wait_for(lambda: self._seq > after_seq, timeout):
                return self._latest
        return None
```

- `self._cond` is a **condition**: a lock plus a doorbell. The lock stops the two threads from touching the slot at the same moment. `notify_all()` rings the bell after a new frame is stored, and `wait_for` sleeps until the bell rings or the timeout passes.
- `self._latest` is the slot. Storing a new frame simply replaces the old one, which is then dropped.
- `seq` counts frames. The main loop asks for a frame newer than the last one it processed, so it never analyses the same frame twice.
- The timestamp comes from `time.perf_counter()`, not `time.monotonic()`. Both clocks only move forward, but on Windows with Python older than 3.13, `time.monotonic()` advances in steps of about 16 ms. We measured 0.015625 s on the test laptop (Python 3.11.9), against 0.0000001 s for `perf_counter`. At 20 frames per second there are only 50 ms between frames, so a 16 ms error would distort every speed calculation. Python 3.13 changed `time.monotonic()` on Windows to the same high-resolution clock ([Python 3.13 notes](https://docs.python.org/3/whatsnew/3.13.html)).

### 12.2 The rate cap

The last lines of the main loop in `main.py`:

```python
            spare = period - (time.perf_counter() - t0)   # rate cap: at most target_fps
            if spare > 0:
                stop.wait(spare)
```

`period` is 1 ÷ `target_fps` = 1 ÷ 20 = 0.05 s, and `t0` is the moment the work on this frame started, just before preprocessing. If MoveNet and the rules took 4 ms, the loop waits 46 ms before it looks for the next frame. Like a **metronome**, the loop allows at most one beat every 50 ms.

Why limit the rate when the laptop could do much more?

- **Leave the CPU for FactoryTalk Optix.** On the device, every inference the service skips is CPU (or NPU) time left for the HMI.
- **20 frames per second is enough.** A swipe lasts roughly 0.3 to 0.6 s, which still gives 6 to 12 samples, enough for the rules in [13](#13-from-keypoints-to-gestures).
- **Less heat** on a small device in a closed enclosure.

`stop.wait(spare)` is used instead of `time.sleep(spare)` because Ctrl+C or a container stop sets the `stop` event and wakes the wait at once, so the service shuts down without delay.

**The side effect: waiting frames.** With a 30 fps webcam, a new frame arrives every 33 ms, but the loop takes one only every 50 ms. A frame that arrives while the loop sleeps waits in the slot, up to one camera frame interval. That is why "frame received → decision" measured 33 to 35 ms at p95 although MoveNet takes only 3 ms. The tip in `config.yaml` follows from this: set the camera itself to about `target_fps`. Frames are then processed as soon as they arrive, and the CPU does not decode frames that are thrown away anyway.

### 12.3 Preparing the image for the model

From `pose.py`:

```python
def center_square(frame):
    """Largest square in the middle of the frame (a view, no copy)."""
    h, w = frame.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return frame[top:top + side, left:left + side]


def preprocess(frame, width, height):
    """BGR camera frame -> uint8 RGB batch of shape [1, height, width, 3]."""
    small = cv2.resize(center_square(frame), (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)[np.newaxis]
```

**Worked example** with a 640 × 480 webcam frame:

1. `side = min(480, 640) = 480`, `top = 0`, `left = (640 − 480) // 2 = 80`. The crop keeps columns 80 to 559: a 480 × 480 square from the middle. "A view, no copy" means NumPy only points into the original array, so the crop itself costs almost nothing.
2. `cv2.resize` shrinks the square by 480 ÷ 192 = 2.5 to 192 × 192 pixels. `INTER_LINEAR` is OpenCV's standard bilinear method: each new pixel is a weighted average of its nearest original pixels.
3. `cv2.cvtColor` swaps BGR to RGB. Doing this on the small picture means 192 × 192 = 36,864 pixels instead of 640 × 480 = 307,200, which is 8.3 times less work.
4. `[np.newaxis]` adds the "batch" dimension that the model expects: shape (192, 192, 3) becomes (1, 192, 192, 3), "one picture".

**Why crop instead of squeezing or padding?**

- **Squeezing** the whole 640 × 480 frame into 192 × 192 would shrink the width by 3.33 and the height by 2.5, so people would look thinner than they are.
- **Padding** (adding borders to make the frame square, as the [TensorFlow tutorial](https://www.tensorflow.org/hub/tutorials/movenet) does) keeps the proportions, but the 640 × 640 padded frame shrinks by 3.33 instead of 2.5, so the operator appears smaller in the model's input.
- **Cropping** keeps the proportions and keeps the operator as large as possible. The price: the left and right 80-pixel strips (a quarter of the width) are invisible to the model. That is acceptable because the operator stands in the middle, and the presence gate ignores people near the edges anyway ([13.2](#132-the-presence-gate)).

A square crop also means x and y share the same unit: 0.1 across is the same distance as 0.1 down. The rules depend on this when they measure distances in shoulder widths ([13.3](#133-body-coordinates-in-shoulder-widths)).

### 12.4 Running MoveNet and reading the result

**Which runtime is loaded.** The top of `pose.py` tries three runtimes in order:

```python
try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        load_delegate = tf.lite.experimental.load_delegate
```

| Runtime | When it is used |
|---|---|
| `ai_edge_litert` (LiteRT) | Inside the container, and in the virtual environment of [23.1](#231-prerequisites): both install `requirements.txt` |
| `tflite_runtime` | The runtime that NXP ships next to its NPU delegate, in case the device image uses it ([19.4](#194-how-the-npu-will-be-used)) |
| `tensorflow` | A computer with only full TensorFlow installed. Our first native Windows test runs used this one, because neither LiteRT nor `tflite_runtime` was installed there |

All three offer the same `Interpreter` class, so the rest of the code does not care which one was found.

**Choosing the NPU or the CPU.** From `PoseEstimator.__init__`:

```python
        if use_npu:
            try:
                self._interpreter = Interpreter(model_path=model_path, num_threads=num_threads,
                                                experimental_delegates=[load_delegate(vx_delegate_path)])
                self._interpreter.allocate_tensors()
                # WHY: the NPU compiles the graph at the first inference, which
                # is where a mismatched driver fails; doing it here keeps the
                # CPU fallback below in charge of that failure too.
                self._interpreter.invoke()
                self.accelerator = "npu"
            except Exception as exc:
                # WHY broad: a missing or mismatched NPU stack fails as OSError,
                # ValueError or RuntimeError depending on the stage; the demo
                # must keep working on the CPU in every case.
                logger.warning("NPU delegate %s unusable (%s); falling back to CPU", vx_delegate_path, exc)
                self._interpreter = None
        if self._interpreter is None:
            self._interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
            self._interpreter.allocate_tensors()
```

- `load_delegate` loads NXP's VX delegate library; `allocate_tensors()` reserves memory for every tensor in the model.
- The extra `invoke()` is a **probe**: one test inference with the NPU. If the delegate loads but fails at its first real use, the failure happens here, inside the `try`, and the CPU takes over. Without the probe, the crash would happen later, on the first camera frame.
- `num_threads` (2 by default) limits how many CPU cores the model may use, leaving the other cores for FactoryTalk Optix.
- After this block, the code checks that the model wants uint8 input (and stops with a clear error otherwise), reads the input size from the model (192 × 192) and logs a line such as `[INFO] pose: MoveNet 192x192 loaded on CPU`.

**One inference:**

```python
    def infer(self, frame):
        """BGR frame -> (17, 3) float32 array of (y, x, score)."""
        self._interpreter.set_tensor(self._input_index, preprocess(frame, self.width, self.height))
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output_index)[0, 0]
```

`set_tensor` copies the prepared picture into the model's input, `invoke()` runs the network, and `get_tensor` reads the output of shape [1, 1, 17, 3]. The `[0, 0]` picks "picture 0, person 0" and leaves a table of 17 rows (keypoints) by 3 columns (y, x, score).

**Warm-up.** Before the camera starts, `warm_up()` runs three inferences on a black picture. The first inferences are slow: on the NPU, the first one compiles the network for the hardware, which takes seconds according to the [Avocado reference](https://github.com/avocado-linux/references/blob/main/imx8mp-npu-pose/README.md). Doing this before the camera starts means the first real gesture is not delayed.

**Reading the numbers.** Suppose the right wrist comes back as `(0.40, 0.30, 0.85)`. In the 480 × 480 crop that is row 0.40 × 480 = 192 and column 0.30 × 480 = 144. Adding the 80 cropped columns, it is pixel x = 224, y = 192 of the original 640 × 480 frame. The service never converts back to pixels, though: the rules work directly with the 0 to 1 values, as the next section shows.

## 13. From Keypoints to Gestures

From here on, no AI is involved. MoveNet has turned the picture into 17 keypoints, and plain Python in `gestures.py` decides whether the wrists made a gesture. The main loop calls `GestureEngine.update(t, keypoints)` once per frame, where `t` is the time the frame arrived, in seconds. The call returns either nothing or one `GestureEvent`: a gesture name and a confidence.

Every call goes through the same steps:

1. If nobody has passed the presence gate for 0.5 s, forget everything ([14.4](#144-forgetting-a-person-or-the-camera)).
2. **Presence gate**: is there a close, centred person? If not, stop here ([13.2](#132-the-presence-gate)).
3. Convert each wrist to **body coordinates** and add it to that wrist's **track** ([13.3](#133-body-coordinates-in-shoulder-widths) and [13.4](#134-a-track-for-each-wrist)).
4. If a gesture fired recently, check whether the **latch** may be released. If not, stop here ([14.1](#141-the-latch-and-its-release)).
5. Run the **detectors** on each wrist that has a new sample ([13.5](#135-swipe-left-and-swipe-right) to [13.7](#137-open-palm)).
6. Pick the **winner**, set the latch and return the event ([13.8](#138-confidence-and-which-gesture-wins)).

The time is passed in as an argument instead of being read from the clock inside. That is what makes the rules testable: a test can play 10 seconds of synthetic poses in a few milliseconds and still get exact timing ([21.1](#211-synthetic-poses-and-injected-time)).

### 13.1 Why rules instead of a second model

A common first idea is to train a second neural network that reads the keypoints and answers "swipe left" or "nothing". We did not, for four reasons:

- **No training data.** A learned classifier needs many labelled recordings of each gesture, made by different people, and even more recordings of everything that is *not* a gesture. This project has none yet.
- **The gestures are geometry.** "The wrist moved at least 1.5 shoulder widths sideways within 0.6 s" can be checked with a ruler and a stopwatch. A model would have to learn what one sentence already says.
- **Rules can be explained and tested.** When a rule fires, you can say exactly which numbers made it fire. The same input always gives the same output, and every rule is covered by unit tests ([21](#21-testing-the-gesture-service)). The thresholds live in `config.yaml` and can change without any retraining ([24](#24-tuning-the-gesture-parameters)).
- **They cost almost nothing.** The rules take tens of microseconds per frame on the CPU and do not use the NPU at all.

The price: every new gesture needs a new hand-written rule, and gestures that are hard to describe with geometry (a circle drawn in the air, finger signs) do not fit this approach. If the list of gestures grows, a small learned classifier on keypoint sequences is the planned next step ([25.4](#254-adding-gestures-in-the-future)).

### 13.2 The presence gate

At a trade show, people walk past behind the operator all the time. MoveNet SinglePose always reports one person, preferring people near the middle of the frame ([10.5](#105-movenet)), but that person may be a visitor three metres back. The **presence gate** decides whether the person in the frame is an operator worth listening to. It looks only at the two shoulders:

```python
    def _gate(self, kps):
        """Return (cx, cy, shoulder_width) of a close, centred person, else None."""
        c = self.config
        ly, lx, ls = kps[LEFT_SHOULDER]
        ry, rx, rs = kps[RIGHT_SHOULDER]
        if min(ls, rs) < c.min_keypoint_score:
            return None
        sw = math.hypot(lx - rx, ly - ry)
        cx, cy = (lx + rx) / 2, (ly + ry) / 2
        if sw < c.min_shoulder_width or abs(cx - 0.5) > c.max_center_offset:
            return None
        return cx, cy, sw
```

| Check | Setting (default) | Passes when |
|---|---|---|
| Shoulders visible | `min_keypoint_score` (0.3) | MoveNet's score is at least 0.3 for *both* shoulders |
| Close enough | `min_shoulder_width` (0.12) | The shoulders are at least 0.12 of the picture apart. In the laptop webcam's 480 × 480 crop, that is 0.12 × 480 = 57.6 pixels |
| Centred | `max_center_offset` (0.35) | The point midway between the shoulders is at most 0.35 from the middle, so x is between 0.15 and 0.85 |

- `math.hypot(dx, dy)` is Pythagoras' theorem, √(dx² + dy²): the straight-line distance between the two shoulders. `cx, cy` is the point midway between them.
- People farther from the camera look smaller, so their shoulders are closer together in the picture. The width check therefore works as a distance limit. How many metres that is depends on the camera's lens, so it must be checked with the real camera ([24](#24-tuning-the-gesture-parameters)).
- When the gate fails, `update()` returns at once and the frame adds nothing to the tracks (`test_far_person_is_gated`, `test_off_center_person_is_gated`).
- `person_present` in the MQTT status ([15.2](#152-payloads)) is true from the first frame that passes the gate until the engine forgets the person, 0.5 s after the last frame that passed.

**A sudden jump.** One more check follows the gate. If the point between the shoulders moved by more than one shoulder width since the last frame that passed the gate, both wrist tracks are cleared:

```python
        if self._center is not None and math.hypot(cx - self._center[0], cy - self._center[1]) > sw:
            self._clear_tracks()
```

An operator standing at the HMI does not move their whole body that far between two frames. A jump that big means MoveNet switched to a different person, or made a mistake. Movement measured across the jump would mean nothing, so the history is thrown away and the new person must start their gesture from the beginning (`test_person_swap_restarts_the_hold`).

### 13.3 Body coordinates in shoulder widths

MoveNet gives positions as fractions of the picture. Using those numbers directly causes two problems:

1. **Walking looks like swiping.** A person who walks across the picture with their arms still also moves their wrists across the picture.
2. **Distance changes the size.** The same hand movement covers twice as much of the picture when the operator stands at half the distance.

The fix is to measure each wrist **relative to the body**, with a ruler that every person carries: the distance between their own shoulders. One **shoulder width (SW)** is the unit, the zero point is the middle of the shoulders, and y points up. From `_add_sample`:

```python
        x = (wx - cx) / sw
        if self.mirror:
            x = -x
        y = (cy - wy) / sw
```

- `wx - cx` is how far the wrist is from the middle of the shoulders, sideways, in picture units. Dividing by `sw` turns it into shoulder widths.
- `cy - wy` is written "backwards" on purpose. Picture y grows downwards ([10.3](#103-what-a-computer-sees-in-an-image)), so this flips it: positive y is above the shoulder line, negative y is below it.
- **Mirror.** The operator faces the camera, so their right hand appears on the *left* side of the picture. `mirror: true` in `config.yaml` ("name gestures from the user's point of view, like a mirror") flips x so that positive x is the operator's own right. A "swipe right" is then a movement to the operator's right. The test `test_mirror_off_reverses_direction` checks the other setting.

**Worked example.** The operator's right shoulder is at x = 0.375 and their left shoulder at x = 0.625, both at y = 0.45. Their right wrist is at x = 0.35, y = 0.40.

1. Shoulder width: `sw` = 0.625 − 0.375 = **0.25**. Middle: `cx` = 0.5, `cy` = 0.45.
2. x = (0.35 − 0.5) ÷ 0.25 = −0.6. The mirror flips it to **+0.6**: the wrist is 0.6 shoulder widths to the operator's right.
3. y = (0.45 − 0.40) ÷ 0.25 = **+0.2**: a fifth of a shoulder width above the shoulder line.

Now the same person takes a step sideways, and everything moves 0.1 to the right in the picture: shoulders at 0.475 and 0.725, wrist at 0.45. The middle is now 0.6, and x = (0.45 − 0.6) ÷ 0.25 = −0.6 again, so +0.6 after the mirror. The wrist did not move in body coordinates, so walking cannot look like a swipe (`test_walking_across_the_image_is_not_a_swipe`).

**The heights that matter.** All the rules use the same vertical scale. From top to bottom:

| y (SW) | What is there |
|---|---|
| +1.4 | `ABOVE_HEAD`, where the tests end a palm-up |
| +1.0 | End line of `palm_up` (`up_end_y`), and top of the swipe band (`swipe_band_max_y`) |
| +0.8 | Top of the open-palm zone (`palm_zone_max_y`) |
| +0.2 | `PALM`, where the tests hold the "stop sign" |
| 0 | The shoulder line |
| −0.2 | `CHEST_L` and `CHEST_R`, where the tests start and end swipes |
| −0.6 | Bottom of the open-palm zone (`palm_zone_min_y`) |
| −1.0 | Release line (`release_y`): a hand below it counts as lowered ([14.1](#141-the-latch-and-its-release)) |
| −1.2 | Bottom of the swipe band (`swipe_band_min_y`) |
| −1.6 | `REST_R` and `REST_L`, the arms hanging down in the tests |

These are reasoned design values. For example, the rules assume that a hand 1 SW above the shoulder line is at about the top of the head. None of these heights has been checked against recordings of real people yet ([25.2](#252-limitations)).

### 13.4 A track for each wrist

A single picture cannot show a movement. The rules therefore keep a **track** for each wrist: a list of its recent positions, like footprints in the snow. Each frame that sees the wrist adds one **sample**:

| Field | Meaning |
|---|---|
| `t` | When the frame arrived, in seconds |
| `x`, `y` | The wrist position in shoulder widths ([13.3](#133-body-coordinates-in-shoulder-widths)) |
| `quality` | The average of the wrist's score and the shoulders' average score, used for the confidence ([13.8](#138-confidence-and-which-gesture-wins)) |
| `forearm_up` | The elbow is visible and the wrist is at least 0.3 SW above it, used by `open_palm` ([13.7](#137-open-palm)) |
| `frame` | The number of the `update()` call that saw it |

A track keeps the last 1.5 s of samples: the longest rule window (1.0 s, for the palm hold) plus 0.5 s. A wrist with a score below 0.3 adds no sample in that frame.

**A movement must be seen to count.** Before a new sample is added, the track is cleared, so it starts again from the new sample, in three cases:

```python
            if (last.frame != self._frame - 1 or dt >= c.max_gap_s
                    or math.hypot(x - last.x, y - last.y) > c.max_wrist_speed * max(dt, 1e-3)):
                track.clear()
```

1. **The previous frame did not see this wrist** (`last.frame != self._frame - 1`). Its score was too low, or the person did not pass the gate.
2. **Too much time passed** since the last sample (`dt >= max_gap_s`, 0.3 s), for example because the camera stalled.
3. **The wrist "teleported"**: it moved faster than `max_wrist_speed`, 15 SW per second. At 20 fps that is 0.75 SW between two frames. `max(dt, 1e-3)` only avoids multiplying by zero.

Why so strict? Imagine that MoveNet sees the right wrist on the left, is unsure for three frames because of motion blur, and then sees it on the right. Did the hand swipe, or did MoveNet mix up the left and right wrists? The rules cannot know, so they do not guess: movement is measured only across consecutive frames that all saw the wrist. Several tests check this, for example a left–right wrist swap (`test_left_right_wrist_swap_glitch_is_not_a_swipe`) and a swipe hidden behind three unsure frames (`test_swipe_across_a_few_missed_frames_is_ignored`).

This has a cost. If a real swipe is so fast that MoveNet loses the wrist halfway, that swipe does not fire either. Whether this happens with real people is one of the things the planned recordings must show ([25.2](#252-limitations)).

The last sample stays in the track until the next one arrives, so a hand that is unseen for a moment still counts as raised. This matters for the latch ([14.1](#141-the-latch-and-its-release)).

After the samples are added, the detectors run on each wrist that got a new sample in this frame.

### 13.5 Swipe left and swipe right

**The rule in words:** within the last 0.6 s, the wrist moved at least 1.5 SW sideways and at most half as much up or down, and it stayed between chest and head height the whole time.

```python
    def _swipe(self, samples):
        c = self.config
        newest = samples[-1]
        # WHY backwards from the newest sample: the first start point far
        # enough away is where the stroke began (the opposite-swipe block is
        # timed from there), and stopping at the first sample outside the band
        # keeps the WHOLE stroke between chest and head, so an arm swinging
        # through hip height is not a swipe.
        for i in range(len(samples) - 1, -1, -1):
            s = samples[i]
            if newest.t - s.t > c.swipe_window_s or not c.swipe_band_min_y <= s.y <= c.swipe_band_max_y:
                return None
            dx, dy = newest.x - s.x, newest.y - s.y
            if abs(dx) >= c.swipe_min_dx and abs(dy) <= c.max_off_axis_ratio * abs(dx):
                return ("swipe_right" if dx > 0 else "swipe_left"), samples[i:]
        return None
```

How to read it:

- The loop walks **backwards** in time, from the newest sample to older ones, and compares each older sample `s` with the newest one.
- It gives up (`return None`) at the first sample that is more than 0.6 s old (`swipe_window_s`) or outside the band from −1.2 to +1.0 SW. So **every** sample of the stroke must be inside the band, not only its two ends. An arm swinging like a pendulum through hip height has both ends in the band but its middle below it, so it is not a swipe (`test_arm_swing_through_hip_height_is_not_a_swipe`).
- It fires at the first sample that is at least 1.5 SW away sideways (`swipe_min_dx`), as long as the vertical difference is at most half the sideways one (`max_off_axis_ratio`, 0.5). A positive `dx` is a movement to the operator's right.
- Walking backwards finds the **newest** possible start point, which is where the stroke really began. This matters for the return-stroke block ([14.3](#143-the-return-stroke)).
- The rule is checked on every frame, so it fires on the first frame that reaches 1.5 SW, even if the hand is still moving.

**Worked example** (`test_swipe_right`): the right wrist waits at `CHEST_L` (x = −0.5, y = −0.2), then moves to `CHEST_R` (x = +1.2) in 0.4 s. At 20 fps that is 8 frames, each one 1.7 ÷ 8 = 0.2125 SW further right.

| Frame of the stroke | x (SW) | Distance from the start | Result |
|---|---|---|---|
| 1 | −0.2875 | 0.2125 | nothing |
| … | … | … | … |
| 7 | +0.9875 | 1.4875 | nothing, just short of 1.5 |
| 8 | +1.2 | 1.7 | `swipe_right` |

Two more numbers:

- The **slowest** swipe that can fire covers 1.5 SW in 0.6 s, an average of 2.5 SW per second. The same 1.7 SW stroke made in 0.8 s covers only about 1.28 SW in any 0.6 s, so it does not fire.
- A stroke of 1.6 SW sideways and 1.0 SW upwards is too diagonal: 1.0 is more than 0.5 × 1.6 = 0.8.

### 13.6 Palm up

**The rule in words:** the wrist is now at least 1 SW above the shoulder line, and within the last 0.7 s it rose by at least 1.2 SW, mostly straight up (sideways movement at most half the rise).

```python
    def _palm_up(self, samples):
        c = self.config
        newest = samples[-1]
        if newest.y < c.up_end_y:
            return None
        for i, s in enumerate(samples):
            rise = newest.y - s.y
            if (newest.t - s.t <= c.up_window_s and rise >= c.up_min_rise
                    and abs(newest.x - s.x) <= c.max_off_axis_ratio * rise):
                return "palm_up", samples[i:]
        return None
```

- The first check is a quick exit: if the newest sample is below the end line (`up_end_y`, 1.0 SW), nothing else is checked.
- The loop walks **forwards** from the oldest sample. It stops at the first sample that is at most 0.7 s old (`up_window_s`), at least 1.2 SW lower than the newest one (`up_min_rise`), and less than half the rise away sideways.

**Worked example** (`test_palm_up`): the right wrist moves from `REST_R` (y = −1.6) to `ABOVE_HEAD` (y = +1.4), a rise of 3.0 SW, in 0.5 s. That is 10 frames of 0.3 SW each.

- Frame 8: y = −1.6 + 8 × 0.3 = 0.8. Below the end line: nothing.
- Frame 9: y = 1.1, above the end line. The hand has risen 2.7 SW since it left the resting position 0.45 s ago, and moved only 0.18 SW sideways. `palm_up` fires, before the hand reaches the top.

Height keeps the similar gestures apart:

- Raising a hand to face height to hold it (`PALM`, y = +0.2) never reaches the end line at +1.0, so it is not `palm_up` (`test_raising_hand_to_hold_is_not_palm_up`).
- A hand held above the head after a `palm_up` is above the open-palm zone, whose top is +0.8, so it does not also fire `open_palm` (`test_palm_up_then_hold_above_head_is_not_open_palm`).
- A hand raised slowly, the same 3.0 SW in 2 s, rises only about 1.05 SW in any 0.7 s, so it fires neither gesture (`test_hand_raised_slowly_above_head_is_neither_gesture`).

### 13.7 Open palm

**The rule in words:** for the last full second, the wrist was seen in every frame, the forearm pointed up, the hand stayed between chest and face height, and it did not wander more than 0.2 SW from its average position.

```python
    def _open_palm(self, samples):
        c = self.config
        newest = samples[-1]
        if newest.t - samples[0].t < c.palm_hold_s:
            return None
        window = [s for s in samples if s.t >= newest.t - c.palm_hold_s]
        if not all(s.forearm_up and c.palm_zone_min_y <= s.y <= c.palm_zone_max_y for s in window):
            return None
        mx = sum(s.x for s in window) / len(window)
        my = sum(s.y for s in window) / len(window)
        if any(math.hypot(s.x - mx, s.y - my) > c.palm_still_radius for s in window):
            return None
        return "open_palm", window
```

1. **Seen for the whole second.** The track must cover at least `palm_hold_s` (1.0 s). A track restarts whenever a frame misses the wrist ([13.4](#134-a-track-for-each-wrist)), so this means "seen in every frame for one second" (`test_open_palm_hold_restarts_after_a_missed_frame`).
2. **Forearm up, inside the zone.** Every sample of the last second must have `forearm_up` and a height between −0.6 and +0.8 SW.
3. **Still.** The code averages the positions of the last second (`mx`, `my`) and checks that no sample is more than 0.2 SW (`palm_still_radius`) from that average. A waving hand fails (`test_waving_hand_is_not_open_palm`); a hand that trembles by up to 0.05 SW passes (`test_jittery_palm_still_fires_once`).

`forearm_up` was computed when the sample was added:

```python
        ey, _, es = kps[elbow]
        forearm_up = es >= c.min_keypoint_score and y - (cy - ey) / sw >= c.palm_min_forearm_rise
```

`(cy - ey) / sw` is the elbow's height in shoulder widths, so the wrist must be at least 0.3 SW (`palm_min_forearm_rise`) above an elbow that MoveNet can see. That is the shape of a "stop" sign. Arms crossed in front of the chest have the wrists hardly higher than the elbows, so they are not a palm (`test_crossed_arms_are_not_open_palm`).

**Worked example** (`test_open_palm_fires_after_hold_time`): the right hand moves from rest to `PALM` in 0.3 s and stays there. `open_palm` fires on the 20th frame after the hand arrives, exactly 1.0 s later. One frame earlier, the 1-second window still contains the last moving sample, at y = −0.1. That sample is about 0.29 SW from the average, more than 0.2, so the hand does not count as still yet.

Remember that this is a **proxy**: it detects "a raised, still hand with the forearm up". It cannot see whether the hand is open or closed ([10.4](#104-pose-estimation-and-keypoints)).

### 13.8 Confidence, and which gesture wins

**Which detector.** For each wrist with a new sample, `_detect` asks the detectors in a fixed order and takes the first answer:

```python
    def _detect(self, track):
        samples = list(track)
        return self._swipe(samples) or self._palm_up(samples) or self._open_palm(samples)
```

The rules are designed so that only one of them can be true at a time: a swipe is mostly sideways, a palm-up is mostly vertical and ends above the head, and an open palm is still and below the head. The order only matters if that design has a gap.

**Confidence.** Each detector also returns the samples it used. The confidence is the average `quality` of those samples, rounded to two decimals:

```python
            confidence = round(sum(s.quality for s in window) / len(window), 2)
```

A sample's quality is the average of the wrist's score and the shoulders' average score. With a wrist score of 0.8 and shoulder scores of 0.9, every sample has quality (0.8 + 0.9) ÷ 2 = 0.85, so the confidence is 0.85 (`test_confidence_is_mean_keypoint_quality`).

> ⚠️ The confidence is **not** the probability that the operator meant the gesture. The rules only answer yes or no. The number says how sure MoveNet was about the points that the rule used, so a low value means a poor view (bad light, a partly hidden arm). The service publishes it but does not filter on it.

**Both hands at once.** If both wrists fire in the same frame, the higher confidence wins. On a tie, the left hand wins, because it is checked first. Only one event is published, and the latch then blocks the other hand ([14.1](#141-the-latch-and-its-release), `test_both_hands_swiping_together_fire_once`).

## 14. One Gesture, One Event

The requirement is easy to state: one gesture must cause one action, even while the operator is still performing it. The rules alone do not guarantee this. Hold a palm still for 10 seconds: in a simulation, the open-palm rule became true after 1 s and stayed true on each of the following 180 frames. Without protection, that would open 181 dialogs.

The protection has three parts: a **latch** after every event ([14.1](#141-the-latch-and-its-release)), **trimming** of the tracks ([14.2](#142-trimming)) and a **block on the return stroke** after a swipe ([14.3](#143-the-return-stroke)).

### 14.1 The latch and its release

**If you know PLCs:** the latch works like a one-shot (ONS) instruction, which passes a single pulse when its input turns on and nothing more until the input has turned off again. Our latch also adds a minimum pause.

When a gesture fires, `update()` stores it in `self._latch`, together with the hand that made it and the time. On every following frame, before any detector runs:

```python
        if self._latch is not None:
            if not self._released(t):
                return None
            self._latch = None
            self._trim_to_newest(*ARMS)   # motion made during the latch never counts
```

```python
    def _released(self, t) -> bool:
        c = self.config
        gesture, side, t_event = self._latch
        if t - t_event < c.cooldown_s:
            return False
        if gesture in OPPOSITE_SWIPE:
            return True
        track = self._tracks[side]
        hand_up = bool(track) and t - track[-1].t < c.lost_s and track[-1].y >= c.release_y
        return not hand_up
```

In words:

1. **Cooldown.** For 1.0 s after any event (`cooldown_s`), nothing fires at all, with either hand.
2. **Swipes** are released as soon as the cooldown is over. (`OPPOSITE_SWIPE` holds the two swipe names, so `gesture in OPPOSITE_SWIPE` means "is a swipe".)
3. **`open_palm` and `palm_up`** also wait until the hand that made them is **lowered** below the release line, 1 SW below the shoulders (`release_y`, −1.0), or has not been seen for 0.5 s (`lost_s`).

The palm gestures wait for the hand to come down because they are *held*: the "stop sign" stays up as long as the operator likes, and the rule stays true the whole time. A swipe is a short movement that ends by itself.

**Example: a palm held for 10 seconds** (`test_palm_held_ten_seconds_fires_once`):

| Time after the hand stops | What happens |
|---|---|
| 1.0 s | `open_palm` fires and the latch is set |
| 1.0 to 2.0 s | Cooldown: nothing can fire |
| 2.0 to 10 s | The cooldown is over, but the hand is still up (y = +0.2, above −1.0): still latched |
| Hand lowered below −1.0 | Latch released, tracks trimmed |

Result: one event. Raising the hand again starts a new 1-second hold, which fires again (`test_palm_refires_after_lowering`). Moving the raised hand around, or losing sight of it for 0.15 s, does not release the latch (`test_palm_moving_while_raised_does_not_refire`, `test_palm_held_through_a_short_dropout_fires_once`).

The whole life cycle:

```mermaid
stateDiagram-v2
  [*] --> NoPerson
  NoPerson --> Ready: a person passes the presence gate
  Ready --> Latched: a rule fires and one event is published
  Latched --> Ready: cooldown over, and for palm gestures the hand is lowered or lost
  Ready --> NoPerson: nobody passes the gate for 0.5 s
  Latched --> NoPerson: nobody passes the gate for 0.5 s
```

### 14.2 Trimming

`_trim_to_newest` throws away every sample of a track except the newest one:

```python
    def _trim_to_newest(self, *sides) -> None:
        for side in sides:
            track = self._tracks[side]
            while len(track) > 1:
                track.popleft()
```

It is called at three moments:

| When | Which tracks | Why |
|---|---|---|
| A gesture fires | Both | The movement that fired can never fire a second time |
| The latch is released | Both | Movement made while the latch was on never counts |
| A return stroke is blocked ([14.3](#143-the-return-stroke)) | That hand's | The blocked stroke cannot fire later |

The newest sample is kept so that the hand's current position stays known. Everything older, and with it every movement made so far, is forgotten. A new gesture must therefore be made in full: a new swipe must cover 1.5 SW again, and a new hold must last a full second again.

### 14.3 The return stroke

**Thought experiment.** Swipe your hand to the right, as if turning a page. Now watch what your hand does next: it comes back to the left. That return movement is a perfect swipe left. Without protection, every "next page" would be followed by "previous page".

The cooldown catches a quick return, within 1 s. A relaxed person may take longer, so there is a second rule: after a swipe, a swipe in the **opposite** direction is blocked if it **starts** less than 1.5 s after the event (`opposite_swipe_block_s`):

```python
    def _blocked(self, gesture, t_start) -> bool:
        if self._last_swipe is None:
            return False
        last_gesture, t_last = self._last_swipe
        # WHY the start time: a slow return stroke that began inside the block
        # time is still the return stroke, even if it ends after it.
        return (gesture == OPPOSITE_SWIPE[last_gesture]
                and t_start - t_last < self.config.opposite_swipe_block_s)
```

`t_start` is the time of the stroke's start point, found by the backward scan in `_swipe` ([13.5](#135-swipe-left-and-swipe-right)). The block uses the start time, not the moment the stroke reaches 1.5 SW, because a slow return stroke can begin inside the 1.5 s and reach the threshold after it. It is still the return stroke.

A blocked stroke also stays blocked **while it goes on**. When a stroke is blocked, `update()` trims that hand's track and remembers the stroke's direction in `_blocked_stroke`:

```python
            if self._blocked(gesture, window[0].t):
                # WHY: consume the natural return stroke after a swipe, now and
                # while it goes on (see _add_sample), so it cannot fire later.
                self._trim_to_newest(side)
                self._blocked_stroke[side] = 1 if gesture == "swipe_right" else -1
                continue
```

Then, for every new sample of that hand, `_add_sample` checks whether the hand still moves the same way:

```python
        direction = self._blocked_stroke[side]
        if direction and len(track) > 1:
            # WHY: a blocked return stroke stays blocked until the hand stops,
            # turns or is lost, even if it goes on past opposite_swipe_block_s.
            if (track[-1].x - track[-2].x) * direction > 0:
                self._trim_to_newest(side)
            else:
                self._blocked_stroke[side] = 0
```

`direction` is +1 for a stroke moving right and −1 for one moving left. The last step multiplied by `direction` is positive only while the hand keeps going the same way. In that case the track is trimmed again, so the stroke can never add up to 1.5 SW. As soon as the hand stops or turns, the block ends. A hand that was not seen for 0.5 s or more has also ended its stroke (`if dt >= c.lost_s: self._blocked_stroke[side] = 0` in `_add_sample`).

Every case is a unit test. After a `swipe_right`, the operator:

| Does this | Result | Why | Test |
|---|---|---|---|
| Moves the hand straight back to the left within 0.4 s | Ignored | Inside the 1 s cooldown | `test_swipe_with_fast_return_fires_once` |
| Swipes left and right again within 0.4 s | Ignored | Inside the cooldown | `test_second_swipe_within_cooldown_is_ignored` |
| Holds still for 1.0 s, then moves 2.0 SW left in 0.3 s | Blocked | The stroke starts 1.0 s after the event | `test_return_stroke_after_cooldown_is_blocked` |
| Holds still for 1.2 s, then moves 2.0 SW left in 0.5 s, reaching 1.5 SW at about 1.6 s | Blocked | Timed from the start of the stroke (1.2 s), not from the moment it reaches 1.5 SW | `test_slow_return_starting_inside_the_block_is_blocked` |
| Holds still for 1.25 s, then moves 3.1 SW left in 0.6 s, until 1.85 s | Blocked to the end | The stroke keeps moving the same way | `test_long_return_stroke_is_blocked_even_past_the_block_time` |
| Does the same, then the wrist is unseen for 1 s, reappears on the right and moves left | `swipe_left` fires | A hand lost for 0.5 s has ended its stroke | `test_swipe_after_the_hand_was_lost_is_not_the_return_stroke` |
| Holds still for 1.6 s, then swipes left | `swipe_left` fires | The stroke starts when the hand leaves the end point, at 1.6 s | `test_opposite_swipe_after_holding_still_past_the_block_fires` |
| Lowers the hand, rests, and starts a full swipe left 2.1 s after the event | `swipe_left` fires | The stroke starts after the block | `test_opposite_swipe_allowed_after_block` |
| Lowers the hand and swipes right again | `swipe_right` fires | Same direction, and the cooldown is over | `test_two_swipes_after_cooldown_fire_twice` |

The row "holds still for 1.6 s" shows why `_swipe` scans backwards. When the left swipe reaches 1.5 SW, its 0.6 s window still contains still samples from 1.3 s onwards. Scanning forwards from the oldest sample would pick 1.3 s as the start, and the swipe would be blocked. Scanning backwards picks the last still sample, at 1.6 s, the moment the hand really started moving.

**Advice for demos:** to go back a page after a "next", hold the hand still for a little more than 1.5 s before swiping the other way. In a simulation at 20 fps, the opposite swipe was blocked after waiting 1.3, 1.45 or 1.5 s at the end point, and fired after waiting 1.55 or 1.7 s.

### 14.4 Forgetting a person or the camera

Three situations wipe the engine's memory:

| Situation | What is forgotten | Where |
|---|---|---|
| Nobody passes the presence gate for 0.5 s (`lost_s`) | Everything: tracks, latch, last swipe, blocked strokes | `reset()`, at the start of `update()` |
| The point between the shoulders jumps by more than one shoulder width ([13.2](#132-the-presence-gate)) | The wrist tracks only; the latch stays | `_clear_tracks()`, in `update()` |
| The camera is lost | Everything | `engine.reset()`, called by `main.py` when the camera state becomes `camera_lost` |

```python
    def reset(self) -> None:
        """Forget everything (camera reconnect, person left)."""
        self._tracks = {side: deque() for side in ARMS}
        self._person_seen: Optional[float] = None
        self._center: Optional[tuple] = None
        self._latch: Optional[tuple] = None        # (gesture, side, t)
        self._last_swipe: Optional[tuple] = None   # (gesture, t)
        self._blocked_stroke = {side: 0 for side in ARMS}   # +1/-1: blocked stroke moving right/left
```

The idea: a gesture half made by someone who has left, or started before a camera outage, must not be completed later by someone or something else (`test_person_lost_mid_gesture_restarts_the_hold`). When only the shoulders jump, the latch stays, so a glitch in the middle of a held palm cannot release it and fire the same gesture a second time.

### 14.5 Known gaps

These are accepted for now and listed again in [25.2](#252-limitations):

- **A quick "back" is ignored.** A deliberate opposite swipe that starts within 1.5 s of a swipe is treated as the return stroke. Worse, its own return movement can then fire the first direction again, so "next, back" becomes "next, next". To the rules, this is exactly the same movement as swipe, return, swipe. The booth guide must tell operators to pause about 1.5 s before reversing.
- **Any sideways arm movement at chest height is a swipe.** Pointing at something or stretching can move a wrist 1.5 SW sideways inside the band, and by the rule's definition that is a swipe. Possible fixes, to be tested with recordings: raise the bottom of the band, or require the forearm to point up.
- **Fast swipes may be lost.** If motion blur hides the wrist in the middle of a swipe, the track restarts and the swipe does not fire ([13.4](#134-a-track-for-each-wrist)).
- **Synthetic tests only.** All the timings in sections 13 and 14 come from computer-generated poses, not from real people.

## 15. MQTT Messages for Gestures

The gesture service never talks to FactoryTalk Optix directly. It publishes MQTT messages to the Mosquitto broker, and FactoryTalk Optix will subscribe to them. If MQTT is new to you, read [2.1](#21-what-is-mqtt) first: it explains publishers, subscribers, the broker and topics. This section shows the exact messages, the MQTT options the service uses, and why each one was chosen. All of it is in `mqtt_client.py`.

### 15.1 Topics

| Topic | Published by | Retained | QoS | When |
|---|---|---|---|---|
| `gesture/paperclip/events` | The gesture service | No | 1 | Once per recognized gesture |
| `gesture/paperclip/status` | The gesture service, or the broker on its behalf ([15.4](#154-retained-status-and-the-last-will)) | Yes | 1 | At start, every 2 s, at once when the camera state changes, and at stop |

- **Naming.** The names follow the pattern of Part 1's topics (`ftoptix/paperclip/sensors`, `anomaly/paperclip/alerts`): who publishes, which machine, what kind of message. The requirement's example topic, `gestures/detected`, is what this project calls `gesture/paperclip/events`. Both names are settings in `config.yaml` (`events_topic` and `status_topic`), so they can change without touching the code.
- **One topic for all gestures.** FactoryTalk Optix needs a single subscription for every gesture there will ever be. A new gesture is a new value inside the message, not a new topic and new HMI configuration ([16.4](#164-adding-a-new-gesture-later)).
- **A separate topic for health.** Events and status have opposite needs: an event must never be replayed, while the latest status must always be available. Two topics let each one get the MQTT options that fit it.
- **Watching both.** Subscribing to `gesture/paperclip/#` receives both topics. In MQTT, `#` is a wildcard that matches everything below that level ([23.7](#237-watch-the-messages)).
- **Client ID.** The service connects as `gesture-recognizer` (`client_id` in `config.yaml`). Client IDs must be unique: when a second client connects with the same ID, the broker disconnects the first one. Never run two copies of the service against the same broker ([23.6](#236-build-and-start-the-container)).

### 15.2 Payloads

Every message is a short JSON text. This is an event:

```json
{"v": 1, "event_id": "0b269db3-c8ea-4852-ab74-80ecda3c01e5", "gesture": "swipe_right", "confidence": 0.87, "timestamp": "2026-10-03T06:25:06.883Z"}
```

| Field | Meaning |
|---|---|
| `v` | Version of the message format, `1` today. A reader can refuse a version it does not know, so a future change of format cannot be silently misread |
| `event_id` | A random **UUID**, a 36-character identifier that is new for every event. The reader uses it to recognize a copy of the same event ([15.3](#153-qos-1-and-duplicates)) |
| `gesture` | `open_palm`, `palm_up`, `swipe_left` or `swipe_right` |
| `confidence` | How sure MoveNet was about the keypoints the rule used, from 0 to 1, rounded to 2 decimals. It is not a probability ([13.8](#138-confidence-and-which-gesture-wins)) |
| `timestamp` | When the service created the event: UTC time with milliseconds, in ISO 8601 format. The final `Z` means UTC. The reader uses it to ignore stale events ([16.2](#162-the-gestureeventmanager-netlogic)) |

The function that builds it:

```python
def make_event(gesture, confidence):
    return {"v": SCHEMA_VERSION, "event_id": str(uuid.uuid4()), "gesture": gesture,
            "confidence": confidence, "timestamp": utc_timestamp()}
```

Why a random ID and not a counter (1, 2, 3, …)? A counter starts again at 1 when the container restarts, and a reader that remembers recent IDs would then throw new events away as "already seen". A random UUID needs no memory of earlier runs.

This is a status message while the camera is lost:

```json
{"v": 1, "state": "camera_lost", "person_present": false, "fps": 0.0, "inference_ms": 0.0, "accelerator": "cpu", "timestamp": "2026-10-03T07:20:57.511Z"}
```

| Field | Meaning |
|---|---|
| `v` | Format version, as above |
| `state` | `starting`, `running`, `camera_lost` or `offline` ([17.1](#171-service-states)) |
| `person_present` | `true` while someone passes the presence gate ([13.2](#132-the-presence-gate)) |
| `fps` | Frames processed per second during the last 2 s, 1 decimal |
| `inference_ms` | Median (p50) time to prepare the image and run MoveNet during the last 2 s, in milliseconds, 1 decimal |
| `accelerator` | `cpu` or `npu`: where MoveNet really runs, after any fallback ([12.4](#124-running-movenet-and-reading-the-result)) |
| `timestamp` | When the status was created, as above |

On the laptop, a `running` status showed `fps` 19.5, `inference_ms` 3.3 and `accelerator` `cpu`.

A reader must accept two forms of `offline`:

- When the service stops cleanly, it sends a normal status with `"state": "offline"` and every field.
- When the broker publishes the last will, the message is only `{"v": 1, "state": "offline"}`, without a timestamp ([15.4](#154-retained-status-and-the-last-will)).

So a reader can count only on `v` and `state`.

The timestamps come from the clock of the computer that runs the service. During development that is the Podman virtual machine, whose clock can differ from the Windows clock that FactoryTalk Optix reads. That is one reason why the planned age check is generous ([16.2](#162-the-gestureeventmanager-netlogic)).

### 15.3 QoS 1 and duplicates

MQTT offers three delivery guarantees, called **QoS** (Quality of Service) levels. Think of the post office:

| QoS | Like | What happens | Packets per message |
|---|---|---|---|
| 0 | A postcard | Sent once. If it is lost, nobody knows | 1 |
| 1 | A registered letter | The sender keeps the message until it gets a receipt, a `PUBACK` packet, and sends it again if needed. If the receipt is lost, the message can arrive twice | 2 |
| 2 | A registered letter with a double handshake | A four-step exchange makes sure the message arrives exactly once | 4 |

The service uses **QoS 1** on both topics:

- **Not 0**, because a lost swipe would leave the operator wondering whether the HMI ignored them, and a lost status change would hide a problem.
- **Not 2**, because it doubles the packets of every message to solve a problem that is cheaper to solve at the receiver. Every event has its own `event_id`, and the planned NetLogic remembers the last 64 IDs and ignores a repeat. A receiver that gives the same result whether a message arrives once or twice is called **idempotent**.

Two MQTT rules matter for the FactoryTalk Optix side:

1. **QoS applies hop by hop.** A message travels from the service to the broker with the QoS the service chose, then from the broker to FactoryTalk Optix with the lower of that QoS and the QoS of the subscription. If the Optix subscriptions were set to QoS 0, events would travel the last hop as postcards. That is why both planned subscriptions use QoS 1 ([16.1](#161-what-the-hmi-will-need)).
2. **Copies can also come from reconnects.** If the connection breaks right after an event was sent, paho (the MQTT library) may send it again after reconnecting, so it can arrive twice, or late. The planned NetLogic handles "twice" with the list of IDs and "late" with the 5 s age check ([16.2](#162-the-gestureeventmanager-netlogic)). An event that is late by less than 5 s can still act once ([25.2](#252-limitations)).

Notice the division of labour. The latch in the service stops the operator's movement from producing two events ([14](#14-one-gesture-one-event)). It cannot help against a copy made by the network after publishing; only the receiver can recognize that, by its ID.

### 15.4 Retained status and the last will

**Retained messages.** Normally the broker forwards a message to the current subscribers and then forgets it. If the publisher sets the **retained** flag, the broker also keeps the latest such message of the topic and hands it to every new subscriber the moment it subscribes. Think of a whiteboard by a door: whoever comes in reads the latest note at once.

- The **status** is retained. When FactoryTalk Optix starts or reconnects, it learns the state of the service immediately instead of waiting up to 2 s for the next heartbeat.
- **Events** are not retained. A retained event would be handed to every new subscriber: restart FactoryTalk Optix, and the last swipe, perhaps an hour old, would change the page again. A status describes the present, so it can be repeated; an event, like a doorbell ring, only means something at the moment it happens.

**Connecting.** `start()` does not wait for the broker:

```python
    def start(self):
        # WHY connect_async + loop_start: the service keeps running, and keeps
        # retrying in the background, even if the broker is not up yet.
        self._client.connect_async(self._host, self._port, keepalive=15)
        self._client.loop_start()
```

paho connects in its own thread ([11.3](#113-inside-the-service-four-threads)) and keeps retrying, waiting between 1 and 10 s between attempts (`reconnect_delay_set(min_delay=1, max_delay=10)`). The camera and the model start even if the broker is not up yet.

**The last will.** At connection time the service also gives the broker a message to publish on its behalf if the service disappears without saying goodbye. MQTT calls it the **Last Will and Testament** (LWT):

```python
        # WHY: if this process dies or its network drops, the broker itself
        # publishes "offline" on the status topic (Last Will and Testament).
        self._client.will_set(status_topic, json.dumps(OFFLINE), qos=1, retain=True)
```

`OFFLINE` is `{"v": 1, "state": "offline"}`. The goodbye is an MQTT `DISCONNECT` packet: the broker publishes the will whenever a connection ends without one. This gives three ways to stop:

| How the service stops | What happens | Measured on the laptop |
|---|---|---|
| Cleanly: Ctrl+C, or `podman compose down`, which sends the container a stop signal | `stop()` publishes a final status with `"state": "offline"`, waits up to 2 s for the broker's receipt, then sends `DISCONNECT`, so the broker throws the will away | ✅ retained `offline` after `podman compose down` |
| Killed or crashed, for example with `podman kill` | No `DISCONNECT`. When a process dies, the operating system closes its connections, so the broker sees the connection end without a goodbye and publishes the will | ✅ `offline` about 1 s after `podman kill` |
| The network disappears silently, for example a pulled cable | Nothing reaches the broker. The service promised a sign of life at least every 15 s (the **keepalive**); the broker waits 1.5 times that, 22.5 s, then gives up and publishes the will | ❌ not measured |

The clean stop in code:

```python
    def stop(self, final_status):
        """Publish a clean final status (normally "offline"), then disconnect."""
        try:
            if self._client.is_connected():
                info = self._client.publish(self._status_topic, json.dumps(final_status), qos=1, retain=True)
                info.wait_for_publish(timeout=2)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Final status not delivered: %s", exc)
        finally:
            self._client.disconnect()
            self._client.loop_stop()
```

The keepalive does **not** catch a frozen main loop: paho's network thread keeps the connection alive even when the main thread hangs. That case belongs to the watchdog ([17.4](#174-internal-watchdog)) and to the planned heartbeat check in FactoryTalk Optix ([16.2](#162-the-gestureeventmanager-netlogic)).

**Coming back.** If the connection was lost long enough for the broker to publish the will, the retained status says `offline` even after paho has reconnected. So on every successful connection, `_on_connect` publishes the last known status again, with a fresh timestamp:

```python
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info("Connected to MQTT broker %s:%d", self._host, self._port)
            # WHY: replaces a retained "offline" left by the last will at once,
            # instead of waiting for the next heartbeat.
            if self._last_status is not None:
                status = dict(self._last_status, timestamp=utc_timestamp())
                client.publish(self._status_topic, json.dumps(status), qos=1, retain=True)
        else:
            logger.error("MQTT connection refused: %s", reason_code)
```

`_last_status` is updated by every `publish_status()` call, even while disconnected, so it always holds the newest status. A broker that answers but refuses the connection, for example because it requires a password, produces the log line `MQTT connection refused`.

### 15.5 Why events are dropped while offline

```python
    def publish_event(self, gesture, confidence):
        """Publish one gesture; returns the payload, or None if dropped."""
        if not self._client.is_connected():
            # WHY drop instead of queue: a gesture delivered seconds late would
            # surprise the operator (FT Optix discards stale events anyway).
            logger.warning("MQTT offline: dropped %s event", gesture)
            return None
        event = make_event(gesture, confidence)
        self._client.publish(self._events_topic, json.dumps(event), qos=1, retain=False)
        logger.info("Gesture %s (confidence %.2f) id=%s", gesture, confidence, event["event_id"])
        return event
```

A gesture is a command meant for right now. Suppose the broker is down for 10 seconds and the operator, seeing no reaction, swipes three times. If the service kept those events and sent them after reconnecting, the HMI would suddenly jump three pages while nobody moves. For an operator interface, dropping is the safer choice: nothing happens, the log says `MQTT offline: dropped swipe_right event`, and the operator repeats the gesture once the HMI responds again.

Status messages are treated differently: the latest one is always kept and sent the moment the connection returns ([15.4](#154-retained-status-and-the-last-will)).

The check runs before the event is handed to paho. An event that paho had already accepted when the connection broke is the reconnect case of [15.3](#153-qos-1-and-duplicates).

### 15.6 Example sequence

FactoryTalk Optix is shown as planned ([16](#16-factorytalk-optix-integration-planned)).

```mermaid
sequenceDiagram
  participant S as Gesture service
  participant B as Mosquitto broker
  participant O as FactoryTalk Optix
  S->>B: CONNECT, with the last will state offline
  S->>B: status starting, retained
  O->>B: SUBSCRIBE to events and status, QoS 1
  B->>O: retained status, sent at once
  S->>B: status running, every 2 s
  B->>O: status running
  S->>B: event swipe_right, QoS 1
  B-->>S: PUBACK
  B->>O: event swipe_right
  Note over S: the webcam stream stops
  S->>B: status camera_lost, at once
  B->>O: status camera_lost
  Note over S: the container is killed, no DISCONNECT
  B->>O: last will, state offline
```

What to notice:

1. FactoryTalk Optix subscribed after the service had published `starting`, and still received it at once, because the status is retained.
2. The event was confirmed with a `PUBACK`: that is QoS 1 between the service and the broker. The broker then runs its own QoS 1 exchange with FactoryTalk Optix (not drawn).
3. A camera state change is reported immediately, without waiting for the 2-second tick.
4. After the kill, the broker speaks for the service.

## 16. FactoryTalk Optix Integration (Planned)

> 📝 **Planned.** Nothing in this section has been built or run yet. It is the design for the FactoryTalk Optix side of the project. The service it relies on is finished ([9.2](#92-what-is-built-and-what-is-not)).

The work splits in two: settings made by hand in FactoryTalk Optix Studio (variables, MQTT subscriptions, pages, a dialog), and the C# code of one NetLogic. The code below is pseudo-code: it shows the logic, not the final C#.

The FactoryTalk Optix terms used here:

| Term | Meaning |
|---|---|
| **Model variable** | A named value in the project's `Model` folder. Screens can show it; logic can read and write it |
| **Dynamic link** | A connection that makes a variable or property follow another variable or property. Change the source, and the linked value follows |
| **`PlainTextSubscription`** | Part of the MQTT client built into FactoryTalk Optix since version 1.6. It receives the messages of one topic and writes the text of each one into a Model variable |
| **NetLogic** | C# code that runs inside FactoryTalk Optix. Its `Start()` runs when it is activated, for example at application startup, and its `Stop()` when it shuts down |
| **NodeId** | The unique identifier of a node: a variable, an object, a screen |
| **PanelLoader** | A container that shows one of several panels at a time. Its `Panel` property, a NodeId, says which one; its `ChangePanel` method switches to another |
| **Session** | One running copy of the HMI user interface. The Web Presentation Engine can have several at once, for example one per connected browser |
| **Dialog** | A popup window, defined as a `DialogType`. `UICommands.OpenDialog` opens it in one session |

### 16.1 What the HMI will need

**Model variables**, in a new folder `Model/Gestures`:

| Variable | Type | Written by | Purpose |
|---|---|---|---|
| `EventJson` | String | MQTT subscription | Text of the latest message on `gesture/paperclip/events` |
| `StatusJson` | String | MQTT subscription | Text of the latest message on `gesture/paperclip/status` |
| `LastGesture` | String | NetLogic | Name of the last accepted gesture, shown on screen |
| `Message` | String | NetLogic | Banner text for `palm_up`; empty when there is nothing to show |
| `ServiceState` | String | NetLogic | The state of the service, shown on screen; `unknown` until the first status arrives |
| `CurrentPage` | NodeId | NetLogic | The page the PanelLoader shows |

**MQTT.** Two `PlainTextSubscription`s on the project's MQTT client, both with **QoS 1** ([15.3](#153-qos-1-and-duplicates)):

- `gesture/paperclip/events` → `Model/Gestures/EventJson`
- `gesture/paperclip/status` → `Model/Gestures/StatusJson`

FactoryTalk Optix and the gesture service must use the **same broker**. The service connects to port 1883 unless `MQTT_PORT` says otherwise. If the project's MQTT client points at another port, start the service with `MQTT_PORT` set to that port ([23.6](#236-build-and-start-the-container)).

**Screens.**

- A folder `UI/Screens/GesturePages` with three demo pages, `Page1`, `Page2` and `Page3`, in this order and easy to tell apart.
- A PanelLoader in `MainWindow` whose `Panel` property is dynamically linked to `Model/Gestures/CurrentPage`.
- A dialog type `GesturePalmDialog` with a title and a Close button.
- A banner label linked to `Message`, and status labels linked to `ServiceState` and `LastGesture`.

**NetLogic.** A runtime NetLogic `GestureEventManager` in the project's `NetLogic` folder. Its own variables are linked to the six Model variables, and two node pointers point at `GesturePalmDialog` and at the `GesturePages` folder.

The idea behind this layout is **state, not commands**. The NetLogic never tells a screen what to do: it changes `CurrentPage`, and the PanelLoader of every session follows the link by itself. There is only one variable to change, and a browser that connects later should show the current page right away.

### 16.2 The GestureEventManager NetLogic

```text
Start():
    find the six variables and the two node pointers
    if anything is missing: log an error and return          (fail visibly)
    EventJson.VariableChange  += OnEvent
    StatusJson.VariableChange += OnStatus
    start a PeriodicTask that runs CheckHeartbeat every 1 s
    if CurrentPage is empty: CurrentPage = first page in GesturePages

OnEvent(text):
    if text is empty: return
    parse the JSON; if that fails: log a warning and return
    if v is not 1, or event_id, gesture or timestamp is missing: log a warning and return
    if event_id is among the last 64 IDs: return              (a QoS 1 copy)
    remember event_id; forget the oldest beyond 64
    if the event is older than 5 s: log a warning and return  (stale)
    LastGesture = gesture
    switch gesture:
        open_palm:   open GesturePalmDialog in every web session where it is not open yet
        palm_up:     Message = "Palm-up gesture received"; log it;
                     a DelayedTask clears Message after 4 s (a new palm_up restarts it)
        swipe_left:  CurrentPage = previous page; before the first comes the last
        swipe_right: CurrentPage = next page; after the last comes the first
        otherwise:   log a warning and ignore it

OnStatus(text):
    parse the JSON; ServiceState = state; remember when it arrived

CheckHeartbeat(), every 1 s:
    if no status arrived for 6 s: ServiceState = "offline (no heartbeat)"

Stop():
    EventJson.VariableChange  -= OnEvent
    StatusJson.VariableChange -= OnStatus
    dispose the PeriodicTask and the DelayedTask
```

Why each choice:

| Choice | Why |
|---|---|
| One NetLogic in the `NetLogic` folder | It belongs to no screen, so one instance sees every event and the duplicate check happens in one place. A NetLogic inside a screen would start again with that screen in every session |
| Subscribing in code with `VariableChange` | The handler gets the new text directly (`e.NewValue`), and there are fewer settings to make in Studio |
| The `-=` lines in `Stop()` | The NetLogic cheat sheet calls them mandatory: without them, the native observer behind the subscription stays alive, and the handler keeps running after the NetLogic has stopped |
| Version check first | A message in an unknown format is ignored, not misread |
| The last 64 IDs | Copies arrive within seconds. With at least 1 s between events (the cooldown, [14.1](#141-the-latch-and-its-release)), 64 IDs cover more than a minute |
| 5 s age limit | A late copy must not act ([15.3](#153-qos-1-and-duplicates)). The limit is generous because during development the service clock (Podman virtual machine) and the FactoryTalk Optix clock (Windows) can differ |
| Dialog only where it is not open yet | Opening twice changes nothing, so a copy that slipped through cannot stack dialogs |
| Wrap-around between pages | The demo never gets stuck on the last page. Stopping at the ends instead is a one-line change |
| Unknown gestures ignored | A newer service with more gestures can run against an older HMI ([16.4](#164-adding-a-new-gesture-later)) |
| `PeriodicTask` and `DelayedTask`, not C# `async`/`await` | The cheat sheet: FactoryTalk Optix tasks may touch project nodes; native `async`/`await` code must not |
| Heartbeat check after 6 s | The status arrives every 2 s, so 6 s means three missed heartbeats. It covers what the last will cannot: if the broker itself, or the network between the broker and FactoryTalk Optix, fails, no will can arrive ([17.5](#175-who-notices-what)) |
| A short `Start()` | The runtime logs a warning when a `Start()` takes more than 2000 ms, and a slow `Start()` delays every NetLogic that starts after it |
| One lock around shared data | The event handler, the `PeriodicTask` and the `DelayedTask` can run at the same time on different threads |
| Fixed constants (5 s, 4 s, 6 s, 64) | Simplicity: nothing to configure in Studio |

Two points must be tested before relying on this design. Each has a fallback:

1. **Does a dynamic link on the PanelLoader's `Panel` property change the page in every session?** Test with two browsers open. Fallback: switch pages with the PanelLoader's `ChangePanel` method from NetLogic, as Rockwell's sample [Optix_Sample_PanelLoaderChangePanelNetLogic](https://github.com/FactoryTalk-Optix/Optix_Sample_PanelLoaderChangePanelNetLogic) shows.
2. **Does `UICommands.OpenDialog` work from this NetLogic's `VariableChange` handler**, with the `UIRoot` of each web session as the parent? The cheat sheet shows exactly this loop over `UI/WebPresentationEngine/Sessions`, but inside an exported method (`[ExportMethod]`), not inside a variable-change handler. Fallback: a small NetLogic inside `MainWindow`, so one per session, that watches a "dialog request" counter in the Model and opens the dialog in its own session.

### 16.3 Example workflow

```mermaid
sequenceDiagram
  actor Op as Operator
  participant S as Gesture service
  participant B as Mosquitto
  participant M as Optix MQTT client
  participant N as GestureEventManager
  participant U as Web sessions
  Op->>S: swipe right
  S->>S: latch, trim the tracks, block swipe_left strokes that start within 1.5 s
  S->>B: event swipe_right, QoS 1
  B->>M: event swipe_right, QoS 1
  M->>N: EventJson changes
  N->>N: check version, ID and age
  N->>U: CurrentPage becomes the next page
  U-->>Op: every browser shows the next page
  Op->>S: the hand moves back to the left
  S->>S: return stroke blocked, no event
```

A demo script that exercises every part:

1. The operator steps onto a floor mark in front of the camera. The status now has `person_present: true`.
2. The operator raises one hand to shoulder or face height, forearm up, and holds it still for 1 s. The dialog opens once, even if the hand stays up.
3. The operator lowers the hand, and the dialog is closed with its Close button.
4. Swipe right, lower the hand, then raise it on the left and swipe right again: Page 1 → Page 2 → Page 3. A swipe made less than 1 s after the previous gesture is ignored (the cooldown, [14.1](#141-the-latch-and-its-release)).
5. Hold the hand still for a little more than 1.5 s, then swipe left: Page 2. A left swipe that starts sooner counts as the return stroke and is ignored ([14.3](#143-the-return-stroke)).
6. Push one hand quickly above the head: the banner shows "Palm-up gesture received" for 4 s.
7. Unplug the camera: the HMI shows `camera_lost`. On the laptop, stopping the webcam bridge produced `camera_lost` after about 2 s. Plug the camera back in: `running`.

### 16.4 Adding a new gesture later

1. **Rule.** Write a detector in `gestures.py` that returns `(gesture name, samples used)` or `None`, and add it to `_detect`. `_detect` asks the detectors in a fixed order and takes the first answer ([13.8](#138-confidence-and-which-gesture-wins)). Design the new rule so that it cannot be true at the same time as an existing one; otherwise the order decides, and one of the two gestures may never fire.
2. **Release.** `_released` treats every gesture that is not a swipe like a palm gesture: it re-arms only after the hand drops below `release_y` ([14.1](#141-the-latch-and-its-release)). A new short movement must be added to the swipe-style release, or it will wait for the hand to drop.
3. **Settings.** Add the new thresholds to `GestureConfig` and to `config.yaml`, with the same names and values. A typo in a key stops the service at startup, and `test_config_yaml_matches_code_defaults` fails if the two disagree.
4. **Tests.** Add synthetic tests: the gesture fires; it fires only once while it is held or continued; similar movements do not fire it ([21](#21-testing-the-gesture-service)).
5. **MQTT.** Nothing changes. The new gesture is a new value of the `gesture` field, and `v` stays 1.
6. **FactoryTalk Optix.** Add one case to the NetLogic's `switch`. Until then, the HMI logs the new gesture and ignores it.

Ideas for new gestures, and for larger extensions, are in [25.4](#254-adding-gestures-in-the-future).

## 17. Health Monitoring and Camera Loss

At a trade show, nobody is standing by to restart a service. Cables get kicked, cameras get unplugged, networks stall. The service is built to recover by itself where it can, and to say what is wrong where it cannot, so the HMI can show it. Six mechanisms work together:

| Mechanism | What it does | Details |
|---|---|---|
| Status messages | Every 2 s, and at once when the camera state changes, the service publishes its state on `gesture/paperclip/status` | [17.1](#171-service-states), [15.2](#152-payloads) |
| Camera reconnect | A lost camera is opened again and again, with growing pauses, until it works | [17.2](#172-camera-loss-and-reconnect) |
| Heartbeat file and healthcheck | The main loop rewrites a file every 2 s; the container runtime checks how old the file is | [17.3](#173-heartbeat-file-and-container-healthcheck) |
| Watchdog | Ends the process if the main loop stops making progress, so the container can be restarted | [17.4](#174-internal-watchdog) |
| Last will | If the service disappears, the broker announces `offline` on its behalf | [15.4](#154-retained-status-and-the-last-will) |
| Heartbeat check in FactoryTalk Optix (planned) | Shows `offline (no heartbeat)` when no status has arrived for 6 s | [16.2](#162-the-gestureeventmanager-netlogic) |

### 17.1 Service states

The `state` field of the status message has four possible values:

| State | Meaning | When |
|---|---|---|
| `starting` | The model is loaded, but no camera frame has arrived yet | Published once right after the model loads, then by the main loop until the first frame arrives or the camera fails |
| `running` | Frames arrive and are processed | Every new frame sets it |
| `camera_lost` | No frames | Opening the camera failed, or a read failed: no frame within the read timeout (2 s for a network stream), the stream or video file ended, or OpenCV raised an error |
| `offline` | The service is not running | The service stopped cleanly, or the broker published the last will |

The first three belong to the camera thread: `camera.py` defines them as `STARTING, RUNNING, CAMERA_LOST`, and the main loop copies the camera's current state into every status. The main loop never sets `offline`; it is the final status of a clean stop, or the last will ([15.4](#154-retained-status-and-the-last-will)).

```mermaid
stateDiagram-v2
  [*] --> starting: model loaded
  starting --> running: first frame
  starting --> camera_lost: open or read fails
  running --> camera_lost: read fails
  camera_lost --> running: frames arrive again
  starting --> offline: clean stop or last will
  running --> offline: clean stop or last will
  camera_lost --> offline: clean stop or last will
  offline --> [*]
```

Two things to notice:

- `starting` never comes back. After a camera failure, the service goes from `camera_lost` straight to `running`.
- `camera_lost` means both "the camera never worked" and "the camera stopped working". The status cannot tell the two apart; the log can, because a camera that worked once logged `Camera opened: …` ([17.2](#172-camera-loss-and-reconnect)).

A video file is a special case, used to replay recordings ([24](#24-tuning-the-gesture-parameters)). At its end, the read fails just like with a lost camera, so the status shows `camera_lost` for about 1 s, and then the file plays again from the start.

### 17.2 Camera loss and reconnect

The camera thread runs this loop until the service stops:

```python
    def _run(self):
        delay = self._reconnect_min_s
        while not self._stop.is_set():
            seq = self._seq
            try:
                cap = self._open_fn(self.source, self._open_timeout_ms, self._read_timeout_ms)
                try:
                    if cap.isOpened():
                        logger.info("Camera opened: %s", redact(self.source))
                        self._read_frames(cap)
                finally:
                    cap.release()
            except Exception:
                # WHY: an unexpected OpenCV error must not end this thread
                # silently (the service would wait for frames forever); it is
                # handled like a lost camera.
                logger.exception("Camera %s failed", redact(self.source))
            if self._seq != seq:
                # WHY: frames arrived, so the camera works: retry quickly,
                # even if this connection then ended with an error.
                delay = self._reconnect_min_s
            if self._stop.is_set():
                break
            self.state = CAMERA_LOST
            logger.warning("Camera %s lost; retrying in %.0f s", redact(self.source), delay)
            self._stop.wait(delay)
            delay = min(delay * 2, self._reconnect_max_s)
```

Step by step:

1. `seq = self._seq` remembers how many frames had arrived before this attempt.
2. `self._open_fn(...)` opens the camera. In the service this is `open_capture`, shown below; the tests pass in a fake camera instead ([21.1](#211-synthetic-poses-and-injected-time)).
3. If the camera opened, the log says `Camera opened: …`, and `_read_frames` reads frames until a read fails or the service stops. Every frame sets the state to `running` ([12.1](#121-keeping-only-the-newest-frame)).
4. `finally: cap.release()` closes the camera in every case, even after an error.
5. `except Exception` catches any unexpected OpenCV error. Without it, the thread would end silently, and the main loop would wait forever for frames that never come. The error is written to the log with its full traceback (`logger.exception`) and then handled like a lost camera.
6. If frames arrived during this attempt, the camera did work, so the pause starts again from the minimum.
7. The state becomes `camera_lost`, the log says, for example, `Camera … lost; retrying in 1 s`, and the thread waits.
8. The pause doubles for the next attempt, up to the maximum.

So the pauses between attempts are 1, 2, 4, 8, 10, 10, 10 … seconds: `reconnect_min_s` is 1 s in the code and `reconnect_max_s` is 10 s in `config.yaml`. This pattern is called **exponential backoff**. Think of calling back a busy phone number: you try again soon, then wait a little longer each time. A camera that is rebooting is not flooded with connection attempts and the log does not fill up with warnings, yet the service never waits more than 10 s before trying again.

The thread waits with `self._stop.wait(delay)`, not `time.sleep(delay)`. When the service shuts down, `stop()` sets the `_stop` event, and the wait ends at once instead of up to 10 s later.

How the camera is opened:

```python
def open_capture(source, open_timeout_ms, read_timeout_ms):
    """Open a webcam index ("0"), a network stream or a video file."""
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms,
    ])
```

- A number such as `0` means a camera attached to the computer that runs the service. OpenCV opens it with its default method.
- Anything else, an `http://` or `rtsp://` address or a file path, is opened through **FFmpeg**, a widely used video library that OpenCV can use as its backend. The two options say how long to wait for the connection (`open_timeout_ms`, 5 s) and for each frame (`read_timeout_ms`, 2 s) before giving up. The OpenCV documentation says that both options apply only to its FFmpeg and GStreamer backends.
- A webcam index gets no timeouts, so how quickly a failing webcam is noticed when the service runs directly on Windows was not measured.

The main loop reacts to every change of the camera state:

```python
            if camera.state != state:
                state = camera.state
                if state == CAMERA_LOST:
                    engine.reset()
                next_status = now          # report camera changes at once
```

- On `camera_lost`, the gesture engine forgets everything ([14.4](#144-forgetting-a-person-or-the-camera)). A gesture started before the outage cannot be completed after it.
- `next_status = now` sends the status in this pass of the loop instead of at the next 2 s tick, so the HMI learns about the change at once.

Camera addresses can contain a user name and a password, such as `rtsp://admin:secret@192.168.1.20/stream`. Every log line passes the address through `redact()` first, which turns it into `rtsp://***@192.168.1.20/stream` ([20.2](#202-the-most-important-lines)).

**Measured on the laptop**, with the webcam bridge as the camera: stopping the bridge produced `camera_lost` after about 2 s. The log then showed new attempts after 1 s and after 2 s. After 30 s without a camera, the service was still running and the container had not been restarted. Starting the bridge again brought the state back to `running`.

### 17.3 Heartbeat file and container healthcheck

Every 2 s, the main loop runs its status block:

```python
            if now >= next_status:
                while recent and recent[0][0] < now - window_s:
                    recent.popleft()
                fps = len(recent) / window_s
                inference_p50 = percentile([r[1] for r in recent], 0.5)
                status = make_status(state, engine.person_present, fps, inference_p50, accelerator)
                publisher.publish_status(status)
                with open(HEARTBEAT_FILE, "w") as f:
                    f.write(status["timestamp"])
                if status_count % LOG_EVERY_N_STATUS == 0:
                    logger.info("state=%s fps=%.1f inference p50=%.1f ms processing p95=%.1f ms "
                                "person=%s accelerator=%s", state, fps, inference_p50,
                                percentile([r[2] for r in recent], 0.95), engine.person_present, accelerator)
                status_count += 1
                next_status = now + window_s
```

- `recent` holds one entry per processed frame: when its processing finished, the MoveNet time, and the processing time from "frame received" to "decision made" ([11.4](#114-one-frames-journey)). Entries older than 2 s (`window_s`, set by `status_interval_s`) are removed first.
- `fps` is the number of frames processed in the last 2 s, divided by 2.
- `inference_p50` is the median time of those frames for preparing the image and running MoveNet. The timed call, `estimator.infer(frame)`, does both.
- The status is published ([15.2](#152-payloads)), and the **heartbeat file** is rewritten.
- Every 15th status, so about every 30 s, the same numbers also go to the log, together with the p95 processing time.

Inside the container, the heartbeat file is `/tmp/gesture_heartbeat`: `tempfile.gettempdir()` returns the system's folder for temporary files, which is `%TEMP%` when the service runs directly on Windows. The file contains the status timestamp, but nobody reads it. What matters is the file's **modification time**, which tells when the main loop last passed this point.

The main loop keeps passing this point while the camera is lost, because waiting for a frame gives up after 0.5 s ([17.4](#174-internal-watchdog)). So the heartbeat means "the main loop is alive", not "the camera works". Reporting the camera is the job of the `state` field.

The healthcheck in `docker-compose.yml`:

```yaml
    healthcheck:
      test: ["CMD", "python", "-c", "import os,sys,tempfile,time; sys.exit(time.time() - os.path.getmtime(os.path.join(tempfile.gettempdir(), 'gesture_heartbeat')) > 10)"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 60s
```

A **healthcheck** is a command that the container runtime runs inside the container from time to time. Exit code 0 means "healthy"; any other exit code counts as a failed check.

- The one-line Python program computes the file's age, `time.time() - os.path.getmtime(...)`, and compares it with 10 s. The comparison gives `True` or `False`. In Python, `True` and `False` behave as the integers 1 and 0, so `sys.exit(True)` ends the program with exit code 1, and `sys.exit(False)` with 0.
- If the file does not exist yet, `getmtime` raises an error. A Python program that ends with an unhandled error also exits with code 1.
- `interval: 10s`: run the check every 10 s. `timeout: 5s`: a check that takes longer than 5 s counts as failed.
- `retries: 3`: three failed checks in a row mark the container **unhealthy**.
- `start_period: 60s`: failed checks during the first 60 s after the container starts do not count, so a slow start is not reported as a problem.

`podman ps` shows the result in its STATUS column, as `(healthy)` or `(unhealthy)`.

Why is the healthcheck in the compose file, and not in the `Dockerfile` where it usually goes? Podman builds images in the **OCI** format by default (OCI is the open standard for container images). That format has no field for a healthcheck, so Podman's build tool, Buildah, warns and discards a `HEALTHCHECK` line in the `Dockerfile`. The compose file attaches the healthcheck when the container is created, which works with both Podman and Docker.

**An unhealthy container is not restarted.** The restart policy `restart: unless-stopped` acts when a container **exits**, and Docker's documentation describes restart policies only for containers that stop or exit, not for unhealthy ones. The health label is information for people and tools. Restarting a stuck service is the watchdog's job ([17.4](#174-internal-watchdog)), and the watchdog is also faster. It ends a stuck process after 15 s, while the healthcheck needs the file to be more than 10 s old and then three failed checks 10 s apart: 30 s or more.

One mismatch must be fixed for the device. The watchdog allows 120 s for startup (`STARTUP_S`), while the healthcheck's start period is only 60 s. If the first NPU compilation on the device takes more than about a minute, the container could be labelled unhealthy before its first heartbeat, although nothing is wrong. Both values should be aligned once the startup time on the device has been measured ([25.2](#252-limitations)).

### 17.4 Internal watchdog

Some failures do not crash anything; they make the program **hang**. A camera driver stops answering, a call into the NPU never returns, a read waits forever. A hung process does not exit, so the restart policy never acts, and the HMI simply stops reacting.

The cure is an old idea from railways, the **dead man's switch**: the driver must keep pressing a pedal, and if the pressure stops, the train brakes by itself. In the service, the main loop "presses the pedal" by calling `feed()`. A separate thread checks every second that this happened recently enough, and if not, it ends the whole process.

```python
class Watchdog(threading.Thread):
    """Exit the process if the main loop stops making progress.

    WHY: container restart policies restart containers that EXIT; a hung
    process (stuck driver, stuck stream) would otherwise stay hung, even
    when marked unhealthy.
    """

    def __init__(self, timeout_s, startup_s):
        super().__init__(name="watchdog", daemon=True)
        self._timeout_s = timeout_s
        self._deadline = time.monotonic() + startup_s

    def feed(self):
        self._deadline = time.monotonic() + self._timeout_s

    def run(self):
        while True:
            time.sleep(1)
            if time.monotonic() > self._deadline:
                logger.critical("No progress before the watchdog deadline; exiting so the container restarts")
                os._exit(1)
```

- `__init__` sets the first deadline `startup_s` seconds ahead: 120 s (`STARTUP_S`). The first inference on the NPU compiles the network, and how long that takes on the i.MX 8M Plus has not been measured yet ([12.4](#124-running-movenet-and-reading-the-result)). The watchdog starts before the model is loaded ([20.2](#202-the-most-important-lines)), so a hang during loading is covered too.
- `feed()` moves the deadline to 15 s (`WATCHDOG_S`) from now.
- `run()` wakes up every second. Once the deadline has passed, it logs a `CRITICAL` line and calls `os._exit(1)`.
- `daemon=True` means that this thread does not keep the program alive when the main thread ends normally.
- `time.monotonic()` is a clock that never jumps backwards, even when the computer's date and time are corrected.

**Why `os._exit(1)` and not `sys.exit(1)`?** `sys.exit()` only raises an exception, and the Python documentation notes that it ends the process only when it is called from the main thread. Called from the watchdog thread, it would end just the watchdog, and the hung main thread would stay hung. `os._exit()` ends the whole process at once, without running any cleanup. Exit code 1 marks the end as a failure.

What should happen next:

1. The process is gone, and the operating system closes its network connection. The broker sees the connection end without a goodbye and publishes the last will, `offline` ([15.4](#154-retained-status-and-the-last-will)).
2. The container has exited, so the restart policy `unless-stopped` starts it again, and the service comes back through `starting`.

Step 1 matches what was measured when the container was killed ([15.4](#154-retained-status-and-the-last-will)). Step 2 follows from the restart policy but has **not been tested**: no real hang was ever produced in the container.

How the main loop feeds the watchdog:

```python
            item = camera.latest(seq, timeout=0.5)
            if item is None:
                # WHY: waiting is normal while the camera (re)connects, but a
                # "running" camera that delivers nothing is stuck: no feed, so
                # the watchdog restarts the service.
                if camera.state != RUNNING:
                    watchdog.feed()
                continue
            watchdog.feed()
```

| Situation | Feed? | Why |
|---|---|---|
| A new frame arrived | Yes | That is progress |
| No frame within 0.5 s, and the camera is `starting` or `camera_lost` | Yes | Waiting is normal while the camera reconnects, and a backoff pause can last 10 s. Without this feed, unplugging the camera would restart the service |
| No frame, although the camera still says `running` | No | Frames stopped while the camera thread believes everything is fine, for example because a read hangs and its timeout did not work. After 15 s, the watchdog ends the process |
| The main loop itself hangs, for example inside `infer` on a stuck NPU driver | Nothing feeds | After 15 s, the watchdog ends the process |

**Known gap:** if opening the camera hangs forever while the state is `starting` or `camera_lost`, the main loop keeps feeding and the watchdog never fires. The status still shows the problem: a `starting` or `camera_lost` that never ends. For network cameras, the 5 s open timeout should prevent such a hang; a webcam index has no timeout ([17.2](#172-camera-loss-and-reconnect)).

⚠️ **How it was verified.** With a throwaway script in which the camera, the model and the MQTT publisher were stand-ins (stubs), and `WATCHDOG_S` was 3 s:

- A stub camera that delivered 10 frames and then nothing, while still reporting `running`, made the process log the `CRITICAL` line and exit with code 1.
- A stub camera reporting `camera_lost` kept the process alive for more than 8 s.

In the real container, 30 s without a camera did not trigger the watchdog ([17.2](#172-camera-loss-and-reconnect)). The 60 unit tests do not cover the watchdog.

### 17.5 Who notices what

| Failure | What the service does | What the HMI will show (planned) | How fast | Status |
|---|---|---|---|---|
| The camera stops delivering frames | Publishes `camera_lost`, forgets any gesture in progress, retries with backoff | `camera_lost` | About 2 s with the webcam bridge | ✅ |
| The camera comes back | Reconnects at its next attempt and publishes `running` | `running` | At the next attempt, at most 10 s later | ✅ |
| The service stops cleanly | Publishes a final `offline` status | `offline` | At once | ✅ |
| The service process is killed | Nothing: it is gone. The broker publishes the last will | `offline` | About 1 s after `podman kill` | ✅ |
| The main loop hangs | The watchdog ends the process; the broker publishes the will; the restart policy should restart the container | `offline (no heartbeat)` after 6 s, then `offline`, then `starting` | 15 s until the exit | ⚠️ watchdog checked with stubs; restart not tested |
| The broker goes down | Keeps running, drops events ([15.5](#155-why-events-are-dropped-while-offline)), keeps the newest status and sends it again after reconnecting | Its own MQTT connection is lost too; `offline (no heartbeat)` after 6 s | paho retries every 1 to 10 s | ⚠️ dropping and resending are unit-tested; the whole case was not tested |
| The network between the service and the broker fails silently | Keeps running; it does not notice at first | `offline (no heartbeat)` after 6 s, `offline` when the broker gives up | The broker publishes the will after 22.5 s | ❌ not measured |
| FactoryTalk Optix restarts | Nothing | The retained status at once. Events are not retained, so no old gesture is replayed to it | At once | 📝 |
| The NPU fails at startup | Falls back to the CPU and reports `accelerator: cpu` | Works the same; MoveNet runs slower on the CPU ([19.3](#193-expected-on-the-imx-8m-plus)) | At startup | ✅ on the laptop, with a simulated failure; never run on the device |

The three ways the service can stop, and what the broker does in each case, are compared in [15.4](#154-retained-status-and-the-last-will).

Two notes on the table:

- On the device, the service and the broker run on the same machine and talk over `127.0.0.1`, so no cable can fail between them. The silent-network row matters when the broker runs on another computer.
- The planned 6 s heartbeat check in FactoryTalk Optix is what makes the hang and network rows visible quickly. Without it, the HMI would keep showing the last status it received, possibly `running`, until the will arrives ([16.2](#162-the-gestureeventmanager-netlogic)).

## 18. Containers and the Laptop Setup

Part 1 explains what a container is ([2.4](#24-what-is-docker-briefly)). This section shows how the gesture service is packaged into one, and how the programs on a Windows laptop find each other.

The laptop runs **Podman** instead of Docker. Podman runs the same images and the same compose files, and its commands mirror Docker's: `podman build`, `podman ps`, `podman compose up`. `podman compose` is a thin wrapper: it runs an external compose program, such as `docker-compose` or `podman-compose`, and connects it to Podman.

Linux containers need a Linux kernel, so on Windows Podman runs them inside a small Linux virtual machine, the **Podman machine**, which it creates with WSL2 (Windows Subsystem for Linux). All measurements in this guide used Podman 5.8.1, with a rootful machine (Podman runs as root inside the virtual machine) that had 10 CPUs and 2 GiB of memory.

### 18.1 The image

`gesture_recognition/Dockerfile`:

```dockerfile
ARG IMAGE_PLATFORM=linux/amd64
FROM --platform=${IMAGE_PLATFORM} python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
# The model is not in Git: run "python tools/fetch_model.py" once before building.
COPY models/ models/

# WHY non-root: the service only opens outgoing connections (camera, broker).
RUN useradd --system gesture
USER gesture

CMD ["python", "src/main.py"]
```

| Lines | What they do | Why |
|---|---|---|
| `ARG IMAGE_PLATFORM=linux/amd64` and `FROM --platform=… python:3.11-slim` | Start from the official slim Python 3.11 image, for 64-bit Intel and AMD processors | A small Debian Linux image with Python already installed. The argument lets the same file build for another processor; the ARM64 build for the device is planned ([25.3](#253-roadmap)) |
| `WORKDIR /app` | Makes `/app` the working folder inside the image | All the following paths are relative to it |
| `COPY requirements.txt .`, then `RUN pip install --no-cache-dir -r requirements.txt` | Installs the Python libraries | Each instruction produces a **layer**, and a layer is rebuilt only when its inputs change. Because the library list is copied **before** the code, a change in the code reuses the slow installation layer. `--no-cache-dir` stops pip from keeping its download cache inside the image |
| `COPY config.yaml .`, `COPY src/ src/`, `COPY models/ models/` | Copies the settings, the code and the model | With the model inside, the image is complete: it can be saved to a file and imported on the device, which then needs no download. The model is not in Git (`.gitignore` excludes `gesture_recognition/models/`), so `tools/fetch_model.py` must run once before the first build; without the folder, the build stops at this line ([23.2](#232-download-the-model)) |
| `RUN useradd --system gesture` and `USER gesture` | Creates an ordinary user and runs the service as that user | The service only opens outgoing connections, to the camera and to the broker, so it needs no administrator rights. Anyone who exploited a flaw in it would get an unprivileged user |
| `CMD ["python", "src/main.py"]` | The command the container runs when it starts | |

`podman images` reports the built image as 466 MB. The base image `python:3.11-slim` accounts for 163 MB and the model for 2.9 MB, so most of the rest is the Python libraries.

The libraries, from `gesture_recognition/requirements.txt`:

```text
# ============================================================================
# requirements.txt — Python dependencies for the gesture recognition service
# ============================================================================
# We keep the list minimal: only what is truly needed.
#
#   paho-mqtt               -> publishes gesture events and status to Mosquitto
#   numpy                   -> arrays passed to and from the TFLite model
#   opencv-python-headless  -> reads RTSP / HTTP-MJPEG / webcam / video files and
#                              resizes frames (headless = no GUI libs, smaller image)
#   pyyaml                  -> reads the config.yaml configuration file
#   ai-edge-litert          -> runs MoveNet (LiteRT, the successor of tflite-runtime);
#                              same package on Windows, amd64 and arm64 Linux
#
# On the edge device the NPU may instead need the tflite_runtime that ships
# with NXP's BSP; src/pose.py uses it automatically when LiteRT is missing.
# ============================================================================

paho-mqtt>=2.0,<3.0
# numpy<2.0: the combination tested with this service.
numpy>=1.26,<2.0
# <4.12: opencv-python-headless 4.12 requires numpy>=2 (see above).
opencv-python-headless>=4.8,<4.12
pyyaml>=6.0,<7.0
ai-edge-litert>=2.2,<3.0
```

- `>=2.0,<3.0` means "version 2.0 or newer, but older than 3.0". By a common convention, a new major version (3.0 here) may change how a library is used, so it is adopted on purpose, not by accident. paho-mqtt 2.0 itself introduced a new form of the callback functions: the one with `reason_code` and `properties` shown in [15.4](#154-retained-status-and-the-last-will).
- numpy stays below 2.0 because that is the combination that was tested, and opencv-python-headless stays below 4.12 because 4.12 requires numpy 2.
- **Headless** means OpenCV without the libraries for drawing windows. The service never opens a window, and the image is smaller.
- `ai-edge-litert` is LiteRT, the runtime that executes MoveNet ([19.2](#192-why-litert)). The **BSP** (Board Support Package) mentioned in the comment is the set of operating system, drivers and libraries that NXP provides for its processors ([19.4](#194-how-the-npu-will-be-used)).

### 18.2 The compose file

The compose file says how to build and run the container. The start of `gesture_recognition/docker-compose.yml`, after the header comments that contain the run recipe:

```yaml
services:
  gesture:
    build:
      context: .
    image: gesture_recognition-gesture:local
    # WHY the default network: Podman creates every compose project network
    # with isolate=true, and two isolated networks cannot reach each other, so
    # a broker started by another stack (e.g. anomaly_detection) would time out.
    network_mode: bridge
    environment:
      - PYTHONUNBUFFERED=1
      - TF_CPP_MIN_LOG_LEVEL=3
      - CAMERA_URL=${CAMERA_URL:-http://host.containers.internal:8090/stream.mjpg}
      - MQTT_HOST=${MQTT_HOST:-host.containers.internal}
      - MQTT_PORT=${MQTT_PORT:-1883}
      - USE_NPU=false
```

The healthcheck comes next ([17.3](#173-heartbeat-file-and-container-healthcheck)). The file ends with:

```yaml
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

| Key | Meaning |
|---|---|
| `build: context: .` | Build the image from the folder that contains the compose file, `gesture_recognition` |
| `image: gesture_recognition-gesture:local` | The name of the built image. `podman images` lists it as `localhost/gesture_recognition-gesture:local` |
| `network_mode: bridge` | Use Podman's default network instead of a separate network for this project ([18.4](#184-networking-between-windows-podman-and-containers)) |
| `PYTHONUNBUFFERED=1` | Python writes every log line at once. Without it, Python collects its output in a buffer when it is not writing to a terminal, and log lines would show up late |
| `TF_CPP_MIN_LOG_LEVEL=3` | Reduces the messages that TensorFlow-based libraries print from their C++ code. Part 1's detector uses the same setting |
| `CAMERA_URL=${CAMERA_URL:-…}` | `${NAME:-default}` takes the variable `NAME` from the shell that runs `podman compose`, or the default written after `:-` if the variable is not set or is empty. This is how `$env:CAMERA_URL` in PowerShell reaches the container |
| `MQTT_HOST`, `MQTT_PORT` | The broker. By default `host.containers.internal:1883` ([18.4](#184-networking-between-windows-podman-and-containers)) |
| `USE_NPU=false` | The laptop has no NPU |
| `restart: unless-stopped` | Start the container again whenever it exits, unless someone stopped it ([17.4](#174-internal-watchdog)) |
| `logging` | With Docker's `json-file` log driver, these options keep at most three log files of 10 MB each, so the logs of a device that runs for months cannot fill its storage |

The four environment variables override the matching values in `config.yaml` ([20.2](#202-the-most-important-lines)).

The default `CAMERA_URL` does not reach a webcam bridge that runs on Windows ([18.4](#184-networking-between-windows-podman-and-containers)). That is why the recipe sets `CAMERA_URL` before starting the container ([23.6](#236-build-and-start-the-container)).

### 18.3 Why the webcam needs a bridge

The gesture container runs inside the Podman machine, and WSL2 does not see the laptop's USB devices. Microsoft documents a way to attach USB devices to WSL2 with an extra open-source tool, usbipd-win, but this project does not use it. The bridge needs no extra installation, and it makes the laptop behave like the production setup, where the camera is a network device.

`tools/webcam_bridge.py` runs with the normal Windows Python. It reads the webcam with OpenCV, compresses every picture to JPEG and serves the pictures as an **MJPEG** stream over HTTP: a single HTTP response that never ends, made of JPEG pictures one after another. Many IP cameras offer the same kind of stream, and a browser shows it as live video.

The response begins with the header `Content-Type: multipart/x-mixed-replace; boundary=frame`. It means: "a sequence of parts, each one replacing the one before; the parts are separated by the line `--frame`". Each picture is then written like this:

```python
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                self.wfile.write(jpeg + b"\r\n")
```

That is the separator, a small header with the picture's type and size in bytes, an empty line, then the JPEG bytes. `\r\n` is the line ending that HTTP uses. On the other side, OpenCV's FFmpeg backend reads this stream like any network camera ([17.2](#172-camera-loss-and-reconnect)).

The options:

| Option | Default | Meaning |
|---|---|---|
| `--device` | `0` | Which webcam: 0 is the first one |
| `--width`, `--height` | `640`, `480` | Requested picture size in pixels |
| `--fps` | `30` | Requested frames per second |
| `--quality` | `80` | JPEG quality, from 1 to 100. Lower values give smaller pictures with more compression artifacts |
| `--bind` | `127.0.0.1` | The network address to listen on ([18.4](#184-networking-between-windows-podman-and-containers)) |
| `--port` | `8090` | The TCP port |

Size and frame rate are only requests. The OpenCV documentation warns that even when setting a property reports success, this "doesn't ensure that the property value has been accepted by the capture device".

How the bridge works inside:

- **One picture, many viewers.** The main thread reads the webcam and puts every new JPEG into a `LatestJpeg` object: the same "newest only" idea as the service's camera slot ([12.1](#121-keeping-only-the-newest-frame)). Each viewer, such as the container or a browser, gets its own HTTP thread, which waits for the next new picture and sends it. Every picture is compressed once, however many viewers there are, and a slow viewer skips pictures instead of falling behind.
- **One address.** Only `/stream.mjpg` exists. Any other path gets the HTTP error 404, "not found".
- **A stalled webcam ends the stream.** If no new picture arrives for 5 s, the bridge closes the response. The service sees a failed read, reports `camera_lost` and reconnects ([17.2](#172-camera-loss-and-reconnect)).
- **A failing webcam ends the bridge.** If the webcam stops delivering pictures, the bridge logs `Webcam stopped delivering frames` and shuts down. You must start it again by hand.
- **DirectShow on Windows.** The code opens the webcam with DirectShow, one of the Windows camera interfaces. Its comment explains why: the webcam opens in about a second, while the default Media Foundation backend can take much longer.
- **Nothing is stored.** Pictures stay in memory only; they are streamed, never saved.
- **No password.** Anyone who can reach the address and the port can watch the webcam. That is why the bridge listens only on `127.0.0.1`, the computer itself, unless told otherwise. The WSL adapter address used with Podman can be reached only from this PC ([18.4](#184-networking-between-windows-podman-and-containers)).

### 18.4 Networking between Windows, Podman and containers

On the laptop, programs run in three kinds of places: on Windows itself, in the Podman machine, and in the containers inside the Podman machine. Each has its own `localhost`. Four rules decide who can reach whom:

1. **Inside a container, `localhost` is the container itself**: not Windows, and not the Podman machine. So a container cannot reach a Windows program at `localhost` or `127.0.0.1`.
2. **Windows reaches published ports at `localhost`.** When a container publishes a port, such as the broker's `"1883:1883"`, Podman forwards it to Windows. FactoryTalk Optix and other Windows programs connect to `localhost:1883`.
3. **A container reaches another container's published port at `host.containers.internal`.** This name points at the Podman machine, where the published ports are. The two containers must also be on networks that are allowed to talk to each other. With Podman 5.8.1 on this laptop, the network that compose created for each project had the option `isolate=true` (`podman network inspect` shows it). Isolated networks cannot reach each other, so a connection from the gesture container to a broker started by another compose project timed out. That is why the gesture service uses `network_mode: bridge`, Podman's default network. Only Podman 5.8.1 was tested.
4. **A container reaches a Windows program at the WSL adapter address.** Windows has a virtual network adapter towards WSL2, named `vEthernet (WSL…)`. A Windows program that listens on that adapter's address, as the bridge does with `--bind`, can be reached from the containers. Only this PC can reach the address, and it can change after a reboot, so the recipe looks it up every time. A connection to a port where nothing listens times out instead of being refused.

Who uses which address:

| From | To | Address | Rule |
|---|---|---|---|
| Gesture container | Webcam bridge on Windows | `http://<WSL adapter address>:8090/stream.mjpg` | 4 |
| Gesture container | Broker container | `host.containers.internal:1883` | 3 |
| FactoryTalk Optix on Windows | Broker container | `localhost:1883` | 2 |
| Gesture service run directly with Windows Python | Broker container | `127.0.0.1:1883`. With `localhost`, connecting took about 2 s longer in our tests | 2 |
| Gesture service run directly with Windows Python | Built-in webcam | `CAMERA_URL=0`; no bridge needed | none |
| Browser on Windows | Webcam bridge | `http://<WSL adapter address>:8090/stream.mjpg`, to check the picture | 4 |

**One port, three places.** The port that the broker publishes on Windows, the `BrokerPort` of the FactoryTalk Optix MQTT client, and the `MQTT_PORT` of the gesture service must be the same number. For example, a broker published as `"1884:1883"` needs `BrokerPort` 1884 and `$env:MQTT_PORT = "1884"`.

None of this applies on the device. There, the containers share the device's own network (`network_mode: host`, [11.1](#111-production-on-the-imx-8m-plus)), and the camera is a real network device.

## 19. Performance and the NPU Plan

Three questions decide whether gesture control feels good on a machine:

- **Latency:** how long from the moment a frame arrives until the decision is made. Short latency makes the HMI feel immediate.
- **Throughput:** how many frames per second the service analyses. With too few, a fast movement falls between two samples.
- **CPU left over:** on the device, the same processor runs FactoryTalk Optix, so every bit of CPU the gesture service uses is taken from the HMI.

Times are given as p50 and p95 ([11.4](#114-one-frames-journey)). **Every measurement in this guide was taken on the laptop. Nothing has run on the i.MX 8M Plus yet.** Laptop numbers do not carry over to the device: it has a much smaller processor, a different instruction set (ARM64 instead of x86-64) and an accelerator the laptop does not have.

### 19.1 Measured on the laptop

| What | Result | How it was measured |
|---|---|---|
| Image preparation + MoveNet, benchmark | p50 2.9 to 3.0 ms, p95 3.8 to 4.3 ms | `tools/bench_pose.py`, 100 runs on a gray 640 × 480 frame, two separate runs ([19.4](#194-how-the-npu-will-be-used)) |
| The same with the older runtime, `tflite-runtime` 2.14 | about 33 ms (p50) | Same laptop, same model ([19.2](#192-why-litert)) |
| Image preparation + MoveNet, live service | p50 3.2 to 3.3 ms | `inference_ms` in the MQTT status, container fed by the webcam bridge |
| Frames analysed per second | 19.5 fps | `fps` in the MQTT status; the rate cap allows at most 20 ([12.2](#122-the-rate-cap)) |
| Frame received → decision made | p95 33 to 35 ms | `processing p95` in the service log; mostly the frame waiting for the rate cap ([12.2](#122-the-rate-cap)) |
| Gesture rules | about 20 to 30 µs per frame | `GestureEngine.update` timed while the 41 gesture tests ran, two runs |
| Model file | 2,894,840 bytes (2.9 MB) | File size, checked by `tools/fetch_model.py` |
| Container image | 466 MB | `podman image ls` ([18.1](#181-the-image)) |
| Webcam stopped → `camera_lost` published | about 2 s | Bridge stopped during a compose run ([17.2](#172-camera-loss-and-reconnect)) |
| Container killed → `offline` will published | about 1 s | `podman kill` on the running container ([15.4](#154-retained-status-and-the-last-will)) |

A **µs** (microsecond) is a thousandth of a millisecond. The rules take about 0.02 to 0.03 ms, roughly a hundredth of the MoveNet time, so they never matter for speed.

What the table says: on the laptop, MoveNet is not the bottleneck. Most of the 33 to 35 ms between "frame received" and "decision made" is waiting for the rate cap, not computing.

### 19.2 Why LiteRT

The runtime is the program that executes the model file ([10.7](#107-cpu-npu-runtime-and-delegates)). Google renamed TensorFlow Lite to **LiteRT** ([announcement](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/)). Two Python packages contain the runtime without the rest of TensorFlow:

| Package | Imported as | Newest release on PyPI when this guide was written | Notes |
|---|---|---|---|
| `tflite-runtime` | `tflite_runtime` | 2.14.0 | The old name |
| `ai-edge-litert` | `ai_edge_litert` | 2.2.0 | The new name. Ready-made wheels for Python 3.11 on Windows (x86-64), Linux (x86-64 and ARM64) and macOS (ARM) |

A **wheel** is a ready-built package file. pip copies it into place instead of compiling source code, so installing takes seconds and needs no compiler. Both packages, and full TensorFlow, offer the same `Interpreter` class, which is why `pose.py` simply tries them in order ([12.4](#124-running-movenet-and-reading-the-result)).

**Why the container uses LiteRT.** On the same laptop with the same model, LiteRT took 3.0 ms (p50) and `tflite-runtime` 2.14 about 33 ms, about 11 times longer. We did not investigate the cause; the measurement decided.

**Natively on Windows.** LiteRT also works outside the container. In a fresh virtual environment, `pip install -r gesture_recognition\requirements.txt` installed LiteRT 2.2.0, and all 60 tests passed with it ([23.1](#231-prerequisites)). Our first native Windows runs, before that test, used full TensorFlow 2.21.0, the third choice in `pose.py`.

**Why `tflite_runtime` is still in the list.** On the device, the runtime must work with NXP's VX delegate. A delegate is compiled code that plugs into the runtime's internals, so NXP builds each delegate together with one runtime:

- NXP's own Python example for the VX delegate imports `tflite_runtime` and loads the delegate with `load_delegate` ([label_image.py](https://github.com/nxp-imx/tflite-vx-delegate-imx/blob/lf-6.18.20_2.0.0/examples/python/label_image.py)).
- A recent NXP Linux release (`lf-6.18.20_2.0.0`) also builds LiteRT 2.1.0, installing the same `ai_edge_litert` package our container uses, plus a LiteRT VX delegate built against it ([NXP LiteRT recipes](https://github.com/nxp-imx/meta-imx/tree/wrynose-6.18.20-2.0.0/meta-imx-ml/recipes-libraries/litert)).

Which runtime the device offers depends on the NXP release it runs, and we do not know that yet ([19.4](#194-how-the-npu-will-be-used)). Our image brings its own LiteRT 2.2.0 from PyPI. Whether NXP's delegate, built for the device's runtime, loads into it must be tested on the device. If it does not, `pose.py` needs no change: it already falls back to `tflite_runtime` when LiteRT is missing. What changes is how the image gets the matching runtime, which is the choice in [19.4](#194-how-the-npu-will-be-used).

### 19.3 Expected on the i.MX 8M Plus

> 📝 **Not measured.** The times in this section are rough planning estimates, not derived from any measurement. NXP's MoveNet page gives no speed figure ([10.5](#105-movenet)).

The chip in short ([10.7](#107-cpu-npu-runtime-and-delegates), [10.8](#108-video-basics)):

| Part | What it is | Role in this project |
|---|---|---|
| CPU | 4 Arm Cortex-A53 cores at up to 1.8 GHz, plus a separate Cortex-M7 real-time core | Linux and all our programs run on the A53 cores, together with FactoryTalk Optix |
| NPU | Rated at 2.3 TOPS | Can run MoveNet's int8 layers through the VX delegate |
| VPU | Hardware video decoder: H.265, H.264, VP9 and VP8 up to 1080p at 60 fps | Not used by the service; the CPU decodes every frame |

| Where MoveNet runs | Estimated time per frame | At `target_fps` 20 (one frame every 50 ms) |
|---|---|---|
| NPU | 10 to 30 ms | Fits in the 50 ms; the CPU still decodes the video and prepares the image |
| CPU, the fallback | 40 to 90 ms | Between about 11 and 20 fps, with MoveNet keeping the CPU busy most of the time, next to FactoryTalk Optix |

**At lower frame rates.** The rules use time windows in seconds, not frame counts (a swipe must happen within 0.6 s), so they should still work at fewer frames per second. This is untested: all tests feed 20 fps. There is a hard floor. When frames are 0.3 s or more apart (about 3.3 fps or fewer), every frame starts a new wrist track (`max_gap_s`, [13.4](#134-a-track-for-each-wrist)), and no gesture can fire.

**The 100 ms goal.** The requirement is a latency below about 100 ms. We count it from "frame received" to "HMI reacts". Two delays are outside it: the gesture itself (holding a palm for 1 s takes 1 s) and the camera's own delay before the frame arrives.

- On the laptop, frame received → decision takes 33 to 35 ms at p95, mostly waiting for the rate cap. Setting the camera to about `target_fps` removes most of the wait ([12.2](#122-the-rate-cap)).
- On the device, MoveNet on the NPU is estimated at 10 to 30 ms. Image preparation on the A53, the MQTT hop and the HMI update have not been measured. An end-to-end measurement is on the roadmap ([25.3](#253-roadmap)).
- A swipe fires as soon as the wrist has moved 1.5 SW, often while the hand is still moving ([13.5](#135-swipe-left-and-swipe-right)). Part of the delay therefore overlaps the end of the movement.

### 19.4 How the NPU will be used

> 📝 **Planned.** The code path and the two tools below exist, but none of them has run on the device yet.

The chain from our code down to the NPU:

```mermaid
flowchart TB
  A["pose.py<br/>LiteRT or tflite_runtime"] --> B["VX delegate<br/>libvx_delegate.so"]
  B --> C["TIM-VX, OpenVX, GAL<br/>user-space libraries from the NXP release"]
  C --> D["galcore kernel driver<br/>/dev/galcore"]
  D --> E["NPU hardware"]
```

Reading it from the top:

- `pose.py` asks the runtime to load the VX delegate, but only when `use_npu` is on (`USE_NPU=true` in the container). Any failure falls back to the CPU ([12.4](#124-running-movenet-and-reading-the-result)).
- The **VX delegate** takes the operations it can run and passes them down. It is built on **TIM-VX**, which VeriSilicon describes as "a software integration module provided by VeriSilicon to facilitate deployment of Neural-Networks on Verisilicon ML accelerators" ([TIM-VX](https://github.com/VeriSilicon/TIM-VX)).
- Below it sit more libraries (OpenVX, GAL and others; the probe below lists eight) that turn the work into commands for the hardware.
- At the bottom is **galcore**, a kernel driver: the part of the operating system that talks to the hardware. NXP's build recipe describes it as the "Kernel loadable module for Vivante GPU" ([recipe](https://github.com/nxp-imx/meta-imx/blob/wrynose-6.18.20-2.0.0/meta-imx-bsp/recipes-kernel/kernel-modules/kernel-module-imx-gpu-viv_6.4.11.p4.6.bb)). Our probe script works on the assumption that the NPU is driven by the same driver and reached through the device file `/dev/galcore`, and checks that first.

**The matching rule.** NXP ships the driver and its user-space libraries as one versioned set. In NXP's build layer, the `galcore` module and the user-space package that provides `libOpenVX` carry the same version number, `6.4.11.p4.6` ([user-space recipes](https://github.com/nxp-imx/meta-imx/tree/wrynose-6.18.20-2.0.0/meta-imx-bsp/recipes-graphics/imx-gpu-viv)). That set belongs to one **BSP** (Board Support Package): the operating system, drivers and libraries NXP provides for its processors. We treat libraries from a different release than the device's driver as unsupported; we have not tried what happens with a mismatch.

A container shares the device's kernel, and therefore its `galcore` driver, but brings its own libraries. So the container's NPU libraries must come from the same release as the device. Three ways to get there:

| Path | How | For | Against |
|---|---|---|---|
| N1 Libraries from the device | Pass `/dev/galcore` into the container and mount the device's NPU libraries read-only | Always the same release as the driver; small image | The libraries may be missing on the device. They were compiled against the device's C runtime and its TFLite or LiteRT version, and both must be compatible with what the container has |
| N2 Image built from the NXP release | Build a minimal image (Python, runtime, TIM-VX, VX delegate) from the same NXP release as the device | Exactly NXP's tested combination | Needs NXP's Yocto build setup, and a rebuild whenever a firmware update changes the device's release |
| N3 CPU only | Leave `use_npu` off | Works today, no coupling | Estimated 40 to 90 ms per frame, and more CPU taken from FactoryTalk Optix ([19.3](#193-expected-on-the-imx-8m-plus)) |

The **C runtime** (glibc and libstdc++) is the basic library almost every Linux program is built on. A library compiled on one system generally needs a C runtime at least as new as the one it was built with. The probe reports both versions, so the N1 risk can be judged before trying it.

**Step 1: take stock of the device.** `tools/npu_probe.sh` only reads information; it installs and changes nothing. Run it on the device as root, for complete output, and keep the report:

```sh
sh npu_probe.sh > npu_probe.txt 2>&1
```

`>` writes the normal output into the file, and `2>&1` sends error messages to the same place. Adding an image name as the first argument also checks that a container started from that image sees `/dev/galcore` (replace `IMAGE` with any image that exists on the device):

```sh
sh npu_probe.sh IMAGE > npu_probe.txt 2>&1
```

| Report section | What it shows | Why we need it |
|---|---|---|
| System | Kernel, OS release, board model, chip ID, CPU count, memory | Identifies the NXP release the device runs |
| NPU kernel driver | Whether `/dev/galcore` exists, the `galcore` version, its last kernel messages | No driver, no NPU |
| NPU user-space libraries | Which of eight libraries are installed (VX delegate, TIM-VX, OpenVX, GAL and four more) | Path N1 needs them on the device |
| TensorFlow Lite | NXP's TFLite folders in `/usr/bin` and the `tflite_runtime` version | Which runtime the delegate was built for |
| C runtime | glibc and libstdc++ versions | Whether the device's libraries can load inside our container |
| Container runtime | Docker version and settings | Whether the device can be passed into a container |
| Temperatures | Every thermal zone, in thousandths of a degree Celsius | A baseline for the heat check ([19.5](#195-bottlenecks)) |

The probe does not yet look for LiteRT, which recent NXP releases install ([19.2](#192-why-litert)). Check it by hand; no output means it is installed, `ModuleNotFoundError` means it is not:

```sh
python3 -c "import ai_edge_litert.interpreter"
```

**Step 2: measure MoveNet.** `tools/bench_pose.py` loads the model exactly as the service does, through `src/pose.py`, so its numbers match what the service will see. From the `gesture_recognition` folder:

```sh
python tools/bench_pose.py
python tools/bench_pose.py --npu
python tools/bench_pose.py --npu --image person.jpg
```

It needs Python with OpenCV, NumPy and a runtime. The image does not contain `tools/`, so on the laptop we ran it inside the gesture image by mounting the folder:

```powershell
$root = (Resolve-Path gesture_recognition).Path
podman run --rm -v "${root}\tools:/app/tools:ro" gesture_recognition-gesture:local python tools/bench_pose.py
```

Output of that run:

```text
accelerator  cpu (2 CPU threads)
model load   30 ms
first run    32 ms
warm runs    p50 2.9 ms, p95 3.8 ms (100 runs)
```

- `accelerator` is the one actually used. With `--npu`, `cpu` here means the NPU failed and the code fell back.
- `model load` includes the NPU graph compilation, which can take seconds, because `pose.py` tests the NPU with one inference.
- `first run` is the first camera-sized frame after loading.
- `warm runs` gives p50 and p95 of image preparation plus MoveNet over 100 runs, on a gray 640 × 480 frame (the laptop stream size).
- With `--image`, it also prints the 17 keypoint scores and their mean. Run once with and once without `--npu`: scores near 0 with the NPU mean the model's quantization does not suit the NPU ([10.6](#106-quantization)).
- Options: `--vx-delegate` (default `/usr/lib/libvx_delegate.so`), `--threads` (default 2, like `num_threads`), `--runs` (default 100), `--model`.

On the laptop, `--npu` falls back cleanly: it prints `accelerator  cpu` and a harmless `AttributeError: 'Delegate' object has no attribute '_library'` message from the runtime, and the benchmark completes. On the device, the container will additionally need `/dev/galcore` and the NPU libraries, depending on the path chosen above.

**Step 3: keep the original quantization first.** A reference project for this chip states that the NPU accelerates per-tensor int8 operations and that per-channel quantization falls back to the CPU. The same project keeps per-channel anyway, because per-tensor collapsed MoveNet's keypoint confidence to about 0 ([Avocado imx8mp-npu-pose](https://github.com/avocado-linux/references/blob/main/imx8mp-npu-pose/README.md)). Our file mixes both kinds ([10.6](#106-quantization)), so part of MoveNet may stay on the CPU even with the delegate. Step 4 shows how much. Re-quantizing to per-tensor would be considered only together with the keypoint-score check of Step 2.

**Step 4: see where each operation runs.** TensorFlow's `benchmark_model` tool can time every operation of the model. The planned command, with flags from the [TFLite benchmark documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/README.md) and NXP's [VX delegate README](https://github.com/nxp-imx/tflite-vx-delegate-imx/blob/lf-6.18.20_2.0.0/README.md):

```sh
benchmark_model --graph=movenet_lightning_int8.tflite --external_delegate_path=/usr/lib/libvx_delegate.so --enable_op_profiling=true
```

- `--graph` is the model file (required).
- `--external_delegate_path` is the delegate library to load.
- `--enable_op_profiling`: "Whether to enable per-operator profiling measurement."

Where `benchmark_model` is installed on the device is not known yet; the probe lists NXP's TFLite folders in `/usr/bin`.

**The graph cache.** The first NPU inference compiles MoveNet for the NPU. The code comments expect this to take seconds; it has not been measured. `warm_up` hides it before the camera starts ([12.4](#124-running-movenet-and-reading-the-result)), but it still delays every start. NXP's examples store the compiled result on disk:

```sh
export VIV_VX_ENABLE_CACHE_GRAPH_BINARY="1"
export VIV_VX_CACHE_BINARY_GRAPH_DIR=$HOME
```

NXP's comment on these lines: "Store on disk .nb files that contains the result of the OpenVX graph compilation", and "This feature is only available for iMX8MPlus to get the warmup time only once" ([common_utils.sh](https://github.com/nxp-imx/nxp-nnstreamer-examples/blob/main/common/common_utils.sh)). Planned: the edge compose file sets both variables and keeps the folder on a named volume, so the cache survives container restarts.

### 19.5 Bottlenecks

> 📝 **Expected, not measured.** The places where the device is likely to be slow, and what is in place or planned for each.

| Bottleneck | Why it matters | In place | Planned |
|---|---|---|---|
| The four A53 cores are shared with FactoryTalk Optix | Every millisecond of CPU the service uses is taken from the HMI | Rate cap at 20 fps ([12.2](#122-the-rate-cap)); color conversion on the 192 × 192 picture only ([12.3](#123-preparing-the-image-for-the-model)) | Measure CPU use on the device with Optix running |
| Every frame is decoded on the CPU | Decoding costs CPU even for frames the rate cap skips; the VPU is unused ([10.8](#108-video-basics)) | Nothing yet | Camera set to 640 × 360 at about `target_fps`. A GStreamer pipeline with NXP's hardware elements only if the CPU budget is missed (NXP's examples use `imxvideoconvert_g2d` for scaling on this chip) |
| The first NPU inference compiles the graph | Delay at every start | `warm_up` before the camera starts | Graph cache on a volume ([19.4](#194-how-the-npu-will-be-used)) |
| Operations the delegate cannot run | Each hand-over between NPU and CPU costs time | Nothing yet | Operation profiling, Step 4 of [19.4](#194-how-the-npu-will-be-used) |
| Per-channel quantization | May run slower on the NPU, or stay on the CPU | Original file kept | Measure; re-quantize only with the accuracy check |
| RTSP buffering | FFmpeg, which OpenCV uses to read RTSP, can buffer frames and add delay | Newest-frame slot ([12.1](#121-keeping-only-the-newest-frame)) | Low-delay options, below the table |
| Shared memory | CPU, NPU, VPU and FactoryTalk Optix use the same main memory | Low resolution, rate cap | 60-minute test with Optix running |
| Heat | A hot chip slows itself down | Rate cap | 60-minute test, reading the thermal zones the probe lists |
| BSP coupling | A firmware update can change the driver and break NPU libraries inside the container | CPU fallback, and `accelerator` in the status ([17.5](#175-who-notices-what)) | Path N1 uses the device's own libraries, which change together with its driver |

**Low-delay options for RTSP (planned).** OpenCV reads RTSP through FFmpeg and passes it extra options from one environment variable, set before the stream is opened:

```text
OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|fflags;nobuffer|flags;low_delay
```

- The format is `key;value` pairs separated by `|` ([OpenCV source](https://github.com/opencv/opencv/blob/4.x/modules/videoio/src/cap_ffmpeg_impl.hpp)).
- `rtsp_transport;tcp` receives the video over TCP, so no pieces go missing, at the cost of possible extra delay on a bad network. OpenCV already prefers TCP when the variable is empty, but setting the variable replaces that default, so the value repeats it.
- `fflags;nobuffer`, in FFmpeg's words: "Reduce the latency introduced by buffering during initial input streams analysis" ([FFmpeg formats](https://ffmpeg.org/ffmpeg-formats.html)).
- `flags;low_delay`, in FFmpeg's words: "Force low delay" ([FFmpeg codecs](https://ffmpeg.org/ffmpeg-codecs.html)).
- The effect has not been measured.

## 20. Gesture Code Walkthrough

This section is a map of the code. [20.1](#201-file-by-file-guide) lists every file and where this guide explains it. [20.2](#202-the-most-important-lines) points to the key lines already shown in earlier sections, then shows the ones that appear nowhere else. All paths are inside the `gesture_recognition/` folder.

### 20.1 File-by-file guide

| File | What it is for | Explained in |
|---|---|---|
| `config.yaml` | Every setting, with comments: camera, model, MQTT and the gesture thresholds | [24](#24-tuning-the-gesture-parameters) |
| `requirements.txt` | The Python packages installed in the image | [18.1](#181-the-image) |
| `Dockerfile` | The recipe for the container image | [18.1](#181-the-image) |
| `docker-compose.yml` | How the container is started on the laptop. Its header comments repeat the run recipe | [18.2](#182-the-compose-file), [23](#23-how-to-run-the-gesture-service) |
| `src/main.py` | Starts everything in the right order, runs the main loop, sends the status, holds the watchdog | [11.3](#113-inside-the-service-four-threads), [12.2](#122-the-rate-cap), [17](#17-health-monitoring-and-camera-loss), [20.2](#202-the-most-important-lines) |
| `src/camera.py` | The camera thread: newest-frame slot, timeouts, reconnect | [12.1](#121-keeping-only-the-newest-frame), [17.2](#172-camera-loss-and-reconnect) |
| `src/pose.py` | Everything about MoveNet: chooses the runtime (LiteRT, then `tflite_runtime`, then TensorFlow), tries the NPU and falls back to the CPU, prepares the image, runs the model, warms it up | [12.3](#123-preparing-the-image-for-the-model), [12.4](#124-running-movenet-and-reading-the-result) |
| `src/gestures.py` | The gesture rules and the repeat protection | [13](#13-from-keypoints-to-gestures), [14](#14-one-gesture-one-event) |
| `src/mqtt_client.py` | Builds and publishes the MQTT messages | [15](#15-mqtt-messages-for-gestures) |
| `tests/test_gestures.py` | 41 tests of the gesture rules | [21](#21-testing-the-gesture-service) |
| `tests/test_pose.py` | 6 tests of image preparation and the model | [21.2](#212-what-the-60-tests-cover) |
| `tests/test_camera.py` | 5 tests of the camera thread | [21.2](#212-what-the-60-tests-cover) |
| `tests/test_mqtt_payloads.py` | 8 tests of the messages and the publisher | [21.2](#212-what-the-60-tests-cover) |
| `tools/fetch_model.py` | Downloads MoveNet and checks its fingerprint | [20.2](#202-the-most-important-lines), [23.2](#232-download-the-model) |
| `tools/webcam_bridge.py` | Laptop only: serves the webcam as an MJPEG stream | [18.3](#183-why-the-webcam-needs-a-bridge) |
| `tools/bench_pose.py` | Times MoveNet on the CPU or the NPU | [19.4](#194-how-the-npu-will-be-used) |
| `tools/npu_probe.sh` | Collects facts about the device's NPU driver and libraries | [19.4](#194-how-the-npu-will-be-used) |
| `models/movenet_lightning_int8.tflite` | The model, 2,894,840 bytes. Downloaded by `fetch_model.py`; the root `.gitignore` keeps the folder out of Git | [10.5](#105-movenet) |

### 20.2 The most important lines

Most key lines were already shown where their idea is explained:

| Lines | File | Shown in |
|---|---|---|
| The newest-frame slot and `latest()` | `src/camera.py` | [12.1](#121-keeping-only-the-newest-frame) |
| The rate cap | `src/main.py` | [12.2](#122-the-rate-cap) |
| Cropping, resizing and color order for the model | `src/pose.py` | [12.3](#123-preparing-the-image-for-the-model) |
| Choosing the runtime, trying the NPU, running MoveNet | `src/pose.py` | [12.4](#124-running-movenet-and-reading-the-result) |
| Presence gate, body coordinates, wrist tracks | `src/gestures.py` | [13.2](#132-the-presence-gate), [13.3](#133-body-coordinates-in-shoulder-widths), [13.4](#134-a-track-for-each-wrist) |
| The three detectors and the choice between them | `src/gestures.py` | [13.5](#135-swipe-left-and-swipe-right) to [13.8](#138-confidence-and-which-gesture-wins) |
| The latch, the return-stroke block, `reset()` | `src/gestures.py` | [14.1](#141-the-latch-and-its-release), [14.3](#143-the-return-stroke), [14.4](#144-forgetting-a-person-or-the-camera) |
| Building an event, connecting, the last will, publishing | `src/mqtt_client.py` | [15.2](#152-payloads), [15.4](#154-retained-status-and-the-last-will), [15.5](#155-why-events-are-dropped-while-offline) |
| The camera's reconnect loop and `open_capture` | `src/camera.py` | [17.2](#172-camera-loss-and-reconnect) |
| The status block of the main loop | `src/main.py` | [17.3](#173-heartbeat-file-and-container-healthcheck) |
| The watchdog, and how the main loop feeds it | `src/main.py` | [17.4](#174-internal-watchdog) |

The seven pieces below appear nowhere else in this guide.

**1. Settings, with four environment overrides** (`src/main.py`):

```python
def load_config(path):
    """Read config.yaml; non-empty environment variables override the
    values that differ between laptop and edge device."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env = os.environ
    if env.get("CAMERA_URL"):
        config["camera"]["url"] = env["CAMERA_URL"]
    if env.get("MQTT_HOST"):
        config["mqtt"]["broker_host"] = env["MQTT_HOST"]
    if env.get("MQTT_PORT"):
        config["mqtt"]["broker_port"] = int(env["MQTT_PORT"])
    if env.get("USE_NPU"):
        config["model"]["use_npu"] = env["USE_NPU"].strip().lower() in ("1", "true", "yes")
    return config
```

- `yaml.safe_load` turns the YAML text into Python dictionaries, lists, numbers and strings. A YAML file can also describe arbitrary Python objects, which the PyYAML documentation calls potentially dangerous for files from untrusted sources. `safe_load` "limits this ability to simple Python objects like integers or lists" ([PyYAML documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)).
- Only four settings can come from environment variables: the ones that differ between the laptop and the device. Everything else lives only in `config.yaml`.
- `env.get(...)` returns `None` for a missing variable and `""` for an empty one. Both count as "not set", so the file's value stays.
- `MQTT_PORT` must be a whole number. A value like `abc` stops the service at startup with `ValueError: invalid literal for int() with base 10: 'abc'`.
- `USE_NPU` means "on" for `1`, `true` or `yes`, in any mix of upper and lower case, with spaces around it ignored. Any other value, for example `on` or `false`, means "off", even when `config.yaml` says `true`.

**2. The start-up order** (`src/main.py`, the beginning of `main()`):

```python
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    config = load_config(os.path.join(ROOT, "config.yaml"))
    cam_cfg, model_cfg, mqtt_cfg = config["camera"], config["model"], config["mqtt"]
    engine = GestureEngine(GestureConfig(**config["gestures"]), mirror=cam_cfg["mirror"])

    publisher = GesturePublisher(mqtt_cfg["broker_host"], mqtt_cfg["broker_port"], mqtt_cfg["client_id"],
                                 mqtt_cfg["events_topic"], mqtt_cfg["status_topic"])
    publisher.start()
    watchdog = Watchdog(WATCHDOG_S, STARTUP_S)
    watchdog.start()
    estimator = PoseEstimator(os.path.join(ROOT, model_cfg["model_path"]), model_cfg["use_npu"],
                              model_cfg["vx_delegate_path"], model_cfg["num_threads"])
    accelerator = estimator.accelerator
    publisher.publish_status(make_status("starting", accelerator=accelerator))
    estimator.warm_up()

    camera = Camera(cam_cfg["url"], cam_cfg["open_timeout_ms"], cam_cfg["read_timeout_ms"],
                    reconnect_max_s=cam_cfg["reconnect_max_s"])
    camera.start()
```

Read it from top to bottom:

1. **Stop requests.** A `threading.Event` is a flag that one thread can set and others can check or wait for. A **signal** is a short notice that the operating system delivers to a running program. `SIGINT` is what Ctrl+C sends. `SIGTERM` is the polite "please stop" that `podman stop` sends first: "By default, containers are stopped with SIGTERM and then SIGKILL after the timeout", and "The default is 10 seconds" ([podman-stop](https://docs.podman.io/en/latest/markdown/podman-stop.1.html)). Both handlers only set the flag. The main loop checks it on every pass and then leaves through the clean shutdown (point 4).
2. **Settings.** `ROOT` is the `gesture_recognition` folder (`/app` in the container), so the paths in `config.yaml` work from any working directory. `GestureConfig(**config["gestures"])` hands every key of the `gestures` section to `GestureConfig` as a named argument; `**` unpacks a dictionary. `GestureConfig` is a **dataclass**, a Python class that mainly holds named values. It has a default for every field and is *frozen*: its values cannot change while the service runs. A key missing from the file keeps the code's default. A misspelled key stops the service at once, for example with `TypeError: GestureConfig.__init__() got an unexpected keyword argument 'palm_hold'`. A unit test keeps the file's values equal to the code's defaults ([21.2](#212-what-the-60-tests-cover)).
3. **MQTT first.** `publisher.start()` returns at once and connects in the background ([15.4](#154-retained-status-and-the-last-will)). The connection is made while the model loads, and a broker that is not up yet does not block the start.
4. **Watchdog before the model.** Its first deadline is 120 s away, so a hang while MoveNet loads, or while it is compiled for the NPU, also ends in a restart ([17.4](#174-internal-watchdog)).
5. **Model.** `PoseEstimator(...)` loads MoveNet. If `use_npu` is on, it tries the NPU with one test inference and falls back to the CPU if that fails ([12.4](#124-running-movenet-and-reading-the-result)). `accelerator` then says which one is used, `npu` or `cpu`.
6. **First status, `starting`.** If the broker connection is not ready yet, the publisher remembers the newest status and sends it as soon as the connection is made ([15.4](#154-retained-status-and-the-last-will)).
7. **Warm-up.** `warm_up()` runs MoveNet three times on a black picture, so the slow first runs happen before the first real frame ([12.4](#124-running-movenet-and-reading-the-result)).
8. **Camera last.** Frames start to flow only when everything that uses them is ready. The main loop follows.

**3. The heart of the main loop** (`src/main.py`):

```python
            frame, t_frame, seq = item
            t0 = time.perf_counter()
            keypoints = estimator.infer(frame)
            t1 = time.perf_counter()
            event = engine.update(t_frame, keypoints)
            t2 = time.perf_counter()
            recent.append((t2, (t1 - t0) * 1000, (t2 - t_frame) * 1000))   # processing: frame received -> decision
            if event is not None:
                publisher.publish_event(event.gesture, event.confidence)
```

- `item` comes from the newest-frame slot ([12.1](#121-keeping-only-the-newest-frame)): the picture, the time the camera thread received it (`t_frame`) and its sequence number. The next pass asks for a frame newer than this `seq`.
- `time.perf_counter()` is Python's high-resolution clock for measuring durations. [22](#22-gesture-decision-log) explains why it is used instead of `time.monotonic()`.
- `estimator.infer(frame)` prepares the picture and runs MoveNet. The result is the 17 keypoints ([12.4](#124-running-movenet-and-reading-the-result)).
- `engine.update(t_frame, keypoints)` applies the gesture rules. The engine never reads a clock itself: it receives the frame's time as an argument. This **injected time** is what makes the tests possible, because a test can feed made-up poses with made-up times, much faster than real time ([21.1](#211-synthetic-poses-and-injected-time)).
- `update` returns one event or `None`, never more than one per frame ([13.8](#138-confidence-and-which-gesture-wins)).
- `recent` keeps three numbers for every frame of the last status interval (2 s): when the frame was finished, how many milliseconds `infer` took, and how many milliseconds passed from "frame received" to "decision made". The status reports the number of entries divided by 2 s as `fps` and the median of the second number as `inference_ms`. The log reports the p95 of the third number as `processing` ([19.1](#191-measured-on-the-laptop)).
- An event is published at once ([15.5](#155-why-events-are-dropped-while-offline)).

**4. The clean shutdown** (`src/main.py`, the end of `main()`):

```python
    finally:
        logger.info("Stopping")
        camera.stop()
        publisher.stop(make_status("offline", accelerator=accelerator))
```

- The main loop sits inside a `try:` block. Python runs the `finally:` part however the `try:` part ends: after a stop request (Ctrl+C or `podman stop`) or after an unexpected error. After an error, the cleanup runs first, and the error then still ends the program.
- `camera.stop()` asks the camera thread to finish and waits for it, at most 5 s.
- `publisher.stop(...)` publishes a final, retained `offline` status, waits up to 2 s for it to be delivered, then disconnects properly. After a proper goodbye, the broker does not send the last will ([15.4](#154-retained-status-and-the-last-will)).
- The watchdog's `os._exit(1)` skips this block on purpose: a hung program cannot be trusted to clean up. Then the broker's last will reports `offline` ([17.4](#174-internal-watchdog)).

**5. Passwords stay out of the logs** (`src/camera.py`):

```python
def redact(source):
    """Hide user:password in camera URLs before they reach the logs."""
    return re.sub(r"//[^/@]+@", "//***@", str(source))
```

- Network cameras usually need a user name and a password, and the URL can carry them: `rtsp://user:password@192.0.2.10/stream`. The service writes the camera URL into its log, for example in `Camera opened: …`. Logs get copied into tickets and chats, so the password must not appear there.
- `re.sub(pattern, replacement, text)` replaces every piece of `text` that matches `pattern`. The pattern is a **regular expression**, a compact language for describing text:
  - `//` matches the two slashes after `rtsp:`.
  - `[^/@]+` matches one or more characters that are neither `/` nor `@`; here, `user:password`.
  - `@` matches the at sign that ends the credentials.
- The result is `rtsp://***@192.0.2.10/stream`. A URL without `@` comes back unchanged, and a webcam number such as `0` becomes the text `"0"`.
- The test `test_credentials_are_redacted` checks three cases: an RTSP URL with credentials, a URL without them, and the webcam number `0` ([21.2](#212-what-the-60-tests-cover)).

**6. A download that cannot leave a broken model** (`tools/fetch_model.py`, inside `fetch()`):

```python
    digest = sha256_of(data)
    if digest != MODEL_SHA256:
        sys.exit(f"Checksum mismatch: expected {MODEL_SHA256}, got {digest}. Nothing saved.")

    target.parent.mkdir(parents=True, exist_ok=True)
    # WHY write-then-rename: an interrupted download never leaves a broken
    # file with the model's name behind.
    partial = target.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(target)
```

- **SHA-256** turns any file into a fingerprint of 64 hexadecimal characters, called a *hash*. Changing even one byte of the file changes the fingerprint completely. The script contains the expected fingerprint, `cd7cc22fa946e5d146a7b98d496853e1923e22828d3972d579973f27f91bb105`, so the laptop, the container and the device are guaranteed to use byte-identical model files. A download that was damaged, or changed on the server, is refused.
- `sys.exit(message)` prints the message to the error output and ends the script with exit code 1 ([Python docs](https://docs.python.org/3/library/sys.html#sys.exit)). At that point, nothing has been written to disk.
- The data is first written to `movenet_lightning_int8.part` and only then renamed to `movenet_lightning_int8.tflite`; `replace` also overwrites an older file. If the script dies while writing, only the `.part` file is left, and the service never finds half a model under the real name.
- Earlier in `fetch()`, if the model is already there with the right fingerprint, the script prints `Model already present: …` and stops. A file with the wrong fingerprint is downloaded again.
- The request sends a browser-like `User-Agent` header. The code comment gives the reason: the address redirects to Kaggle storage, which may reject requests that do not look like a normal client.

**7. Opening the webcam on Windows** (`tools/webcam_bridge.py`, inside `main()`):

```python
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.device, backend)
    if not cap.isOpened():
        sys.exit(f"Cannot open webcam {args.device} (in use by another app?)")
```

- `sys.platform` is `"win32"` on Windows ([Python docs](https://docs.python.org/3/library/sys.html#sys.platform)), also with 64-bit Python: the laptop's Python 3.11.9 reports `win32`. On Windows the bridge asks OpenCV for DirectShow ([18.3](#183-why-the-webcam-needs-a-bridge) explains why). Elsewhere, `CAP_ANY` lets OpenCV choose.
- `isOpened()` is `False` when the webcam could not be opened. The script then stops with a message that suggests a likely cause: another program is using the webcam ([23.9](#239-troubleshooting)).

## 21. Testing the Gesture Service

A **unit test** is a small program that runs one piece of the real code with known input and checks the result. The gesture service has 60 of them, written with Python's built-in `unittest` module. On the laptop they all run in under a second, without a camera, a broker or a person. How to run them is in [23.3](#233-run-the-unit-tests).

Every test file first puts the `src` folder on Python's search path, so that `import gestures` finds the real code:

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
```

### 21.1 Synthetic poses and injected time

The gesture rules are the hardest part to get right. Testing them with people in front of a camera would be slow, and no movement can be repeated exactly. So `tests/test_gestures.py` makes up poses, called **synthetic** poses, and feeds them to the real `GestureEngine`. Four ideas make this work.

**Idea 1: name hand positions in body coordinates.** Positions are (x, y) pairs in shoulder widths (SW) from the middle of the shoulders, with y pointing up and +x to the user's right. These are the coordinates the rules use ([13.3](#133-body-coordinates-in-shoulder-widths)):

```python
REST_R = (0.6, -1.6)       # right hand hanging by the side
REST_L = (-0.6, -1.6)
PALM = (0.4, 0.2)          # "stop sign" in front of the right shoulder
CHEST_L = (-0.5, -0.2)     # right hand across the body, chest height
CHEST_R = (1.2, -0.2)      # right hand out to the right, chest height
ABOVE_HEAD = (0.4, 1.4)
```

**Idea 2: turn body coordinates into what MoveNet would output.** `make_pose(...)` builds the 17 × 3 array of (y, x, score) that MoveNet returns ([10.5](#105-movenet)). It fills in the shoulders, elbows and wrists and leaves every other keypoint at zero. Its parameters change what MoveNet would "see":

| Parameter | Default | Used for |
|---|---|---|
| `right`, `left` | `REST_R`, `REST_L` | Where the hands are |
| `right_elbow`, `left_elbow` | 0.8 SW below the wrist, so the forearm points up | Crossed arms, and a forearm too low to count as a raised palm |
| `wrist_score`, `body_score` | 0.9 | Unsure or missed keypoints, and a person who does not pass the presence gate |
| `center` | (0.5, 0.45) | Where the middle of the shoulders is in the square picture the model sees, as fractions of its width and height |
| `sw` | 0.25 | How wide the shoulders appear, as a fraction of the picture. A smaller value is a person further away |
| `mirror` | `True` | Whether the picture is mirrored, like `camera.mirror` |

`right_hand(positions)` turns a list of positions into a list of poses with the right hand at each position. Extra settings, such as `sw=0.08`, are passed on to `make_pose`.

**Idea 3: a movement is a list of positions, 20 per second.** `move(start, end, seconds)` returns positions along a straight line, one per frame at 20 fps. The start itself is left out, because the previous movement already ended there. `stay(position, seconds)` repeats one position. Adding lists strings movements together. A swipe to the right:

```python
def swipe_right_motion():
    return move(REST_R, CHEST_L, 0.4) + stay(CHEST_L, 0.3) + move(CHEST_L, CHEST_R, 0.4)
```

The hand comes up across the body in 0.4 s, pauses for 0.3 s, then sweeps 1.7 SW to the right in 0.4 s.

**Idea 4: injected time.** The engine never reads a clock; every call passes in the frame's time ([20.2](#202-the-most-important-lines), point 3). The tests' `Session` keeps its own clock, which starts at 100.0 s and moves forward 1/20 s per pose:

```python
class Session:
    """Feeds frames at FPS into one engine and records the events."""

    def __init__(self, config=None, mirror=True):
        self.engine = GestureEngine(config or GestureConfig(), mirror=mirror)
        self.t = 100.0
        self.events = []

    def play(self, poses):
        for kps in poses:
            self.t += DT
            event = self.engine.update(self.t, kps)
            if event is not None:
                self.events.append(event)
        return self

    def rest(self, seconds=0.5):
        return self.play(right_hand(stay(REST_R, seconds)))

    @property
    def names(self):
        return [e.gesture for e in self.events]
```

No gesture test waits for a real clock. The 41 gesture tests replay 2,525 frames, about 126 seconds of movement, in about 0.06 s on the laptop.

**Reading a test.** One complete test:

```python
    def test_swipe_with_fast_return_fires_once(self):
        s = Session().rest().play(right_hand(swipe_right_motion()))
        s.play(right_hand(move(CHEST_R, CHEST_L, 0.4) + move(CHEST_L, REST_R, 0.4) + stay(REST_R, 2.0)))
        self.assertEqual(s.names, ["swipe_right"])
```

Read it as a story. The operator stands with the hands down for 0.5 s (`rest()`), then swipes right. The hand travels straight back across the body in 0.4 s, which is itself a fast movement to the left, then drops to the side and rests there for 2 s. The test expects exactly one event, `swipe_right`: the return stroke must not fire `swipe_left` ([14.3](#143-the-return-stroke)). If the two lists differ, `assertEqual` fails the test and prints both.

**Fakes for the other modules.** The camera, MQTT and NPU tests replace the outside world with small stand-ins written in the test files, usually called **fakes**:

- `FakeCapture` stands in for OpenCV's `VideoCapture`. It delivers a chosen number of tiny 4 × 4 pixel frames, each filled with its frame number. After that every read fails; if the test asks for it, the read first waits until the test lets it go. A `FakeCapture` can also play a camera that cannot be opened at all. `Camera` accepts the function that opens the stream as a parameter, `open_fn`, so a test can hand it fakes instead of real cameras. The tests also shorten the reconnect pauses to a few hundredths of a second.
- `FakeClient` stands in for paho's MQTT client. It records the last will, every publish, and every connect, loop and disconnect call, so a test can check exactly what would have been sent. The test decides whether the fake is "connected".
- `unittest.mock.patch.object` replaces a name inside a module for the duration of one `with` block. `test_pose.py` uses it to swap in an interpreter that raises `RuntimeError("graph compilation failed")` at its first inference on the NPU, and a `load_delegate` that pretends to load the NPU library. That is how the NPU fallback is tested on a laptop that has no NPU.

### 21.2 What the 60 tests cover

| File | Group | Tests | What they check |
|---|---|---|---|
| `test_gestures.py` | `TestRecognition` | 7 | Each of the four gestures fires when performed properly. `open_palm` fires only after the hold time, and the hold starts again after a single frame in which the wrist was not seen clearly. With the mirror off, the same movement is the opposite swipe. The confidence is the mean keypoint score: wrist scores of 0.8 and body scores of 0.9 give 0.85 ([13.8](#138-confidence-and-which-gesture-wins)) |
| | `TestNoRepeatActivation` | 17 | One movement gives one event. A palm held for 10 s fires once. So does a palm whose wrist is not seen for a moment, or one that moves to another spot and is held there. A palm fires again only after the hand was lowered. Raising the hand to the hold position is not `palm_up`; holding it above the head after `palm_up` is not `open_palm`; a hand raised slowly above the head is neither. Return strokes do not fire: fast, slow, long, or after the cooldown. A swipe after the hand was lost is a new swipe, not a return stroke. A second swipe within the cooldown is ignored; two swipes after it fire twice. An opposite swipe fires after the block time, also after the hand held still. Both hands swiping together fire once |
| | `TestFalsePositives` | 12 | Movements that must not fire: walking across the picture; an arm swinging at hip height, or through hip height like a pendulum; low wrist scores; a swipe or a rise that happened while the wrist or the person was not seen clearly; crossed arms; the model swapping the left and right wrist for a moment; a waving hand. A palm with random jitter of up to ±0.05 SW still fires exactly once |
| | `TestPresence` | 4 | The presence gate ([13.2](#132-the-presence-gate)): a person too far away (`sw=0.08`) or off-center (`center` x at 0.9) is ignored. Another person stepping in (the shoulder center jumps 1.2 SW) or losing the person restarts a palm hold |
| | `TestConfigFile` | 1 | The `gestures` values in `config.yaml` equal the code defaults, so the tests check the deployed thresholds |
| `test_pose.py` | `TestPreprocessing` | 3 | The center square of a landscape and of a portrait picture. The model input is one 192 × 192 RGB picture of `uint8` values, and a blue pixel stays blue after the BGR-to-RGB swap ([12.3](#123-preparing-the-image-for-the-model)) |
| | `TestRealModel` | 3 | With the real model file: CPU inference returns 17 keypoints with values from 0 to 1. A missing NPU library falls back to the CPU with a warning; so does an NPU that fails at its first inference. Skipped when the model has not been downloaded |
| `test_camera.py` | `TestCamera` | 5 | `latest()` returns only the newest frame. A lost camera is opened again, with the state `camera_lost` meanwhile. An error inside a read is handled like a lost camera. The reconnect pause starts again from the minimum after frames arrived. Passwords are hidden ([20.2](#202-the-most-important-lines), point 5) |
| `test_mqtt_payloads.py` | `TestPayloads` | 3 | An event has exactly the five fields, a valid UUID and a UTC timestamp with milliseconds. Two events get different IDs. The status rounds its numbers to one decimal |
| | `TestPublisher` | 5 | The last will is a retained `offline`. Events are dropped while disconnected. Events use QoS 1 and are not retained. The status is retained and sent again after a reconnect. Stopping sends the final status, then disconnects |

**Not covered by the unit tests:**

- The watchdog thread. It was checked separately with stand-ins ([17.4](#174-internal-watchdog)).
- `main()`, which wires the pieces together. It was checked by running the service in its container with the webcam bridge ([19.1](#191-measured-on-the-laptop)).
- The webcam bridge. It was checked by running it.
- Real people. The thresholds are reasoned design values, not yet tuned on recordings ([25.2](#252-limitations)).

### 21.3 Test-first and the mutation check

**Test-first.** The gesture rules were written test-first, an approach also called *test-driven development*. The first run of `test_gestures.py` failed with an `ImportError`, as it should, because `gestures.py` did not exist yet: the "red" step. Then the code was written until all 25 tests of that first version passed: the "green" step. Seeing a test fail first proves that it can fail. A test that has never failed may pass for the wrong reason, for example because it never reaches the code it is meant to check.

**The mutation check.** Passing tests do not prove that the tests are strict enough. So the code was broken on purpose, one small change at a time: a threshold made so loose that it never stops anything, or a guard skipped. Each broken version is called a **mutant**. After each change the tests ran again. A mutant that no test notices "survives", and points to a line the tests do not really protect.

The first runs of this check found gaps, which led to four new tests, for example a second swipe within the cooldown and a waving hand. In the run with 14 mutants, 12 were caught. The two survivors showed weak tests, not wrong code:

| Mutant | Why it survived | Fix |
|---|---|---|
| `opposite_swipe_block_s` set to 0, so no return-stroke block | The return stroke in the test was too short to fire anyway. After the trim at the latch release ([14.2](#142-trimming)), it covered only 1.4875 SW, just under the 1.5 SW a swipe needs | The test's return stroke now covers 2.0 SW instead of 1.7 SW, so only the block can stop it |
| `palm_zone_max_y` set to 100, so the hold zone has no upper limit | The test that holds a hand above the head for 3 s first fires `palm_up`. After that, the latch blocked `open_palm`, not the zone | New test: a hand raised slowly above the head is neither gesture. The rise is too slow for `palm_up`, so nothing but the zone stops `open_palm` |

The check also runs the tests on the unbroken code, as a reference. There, one of the four new tests failed because the test itself was wrong: it moved the person by exactly 1.0 SW, but the rule reacts only to *more than* 1 SW. It now uses 1.2 SW. After these changes, all 14 mutants were caught.

**Review rounds.** Two code reviews found real problems; the decision log lists the fixes ([22](#22-gesture-decision-log)). Every fix followed the same pattern: first a new test that reproduces the problem, then a run that shows it failing for the expected reason, then the fix.

- **Round 1:** 6 new tests failed before the fixes: 4 for the gesture rules, 1 for the camera and 1 for the NPU fallback. A seventh new test, a **guard**, passed before and after. A guard pins down behavior that must not change while the code around it is fixed. A mutant that picks the oldest possible start of a swipe instead of the newest was caught by this guard.
- **Round 2:** 5 new tests failed before the fixes, and another guard passed. One more problem turned up during the fixes and got its own failing test first, which brought the total to 60. A mutation check of the new gesture code followed: each mutant was caught only by the tests meant to catch it.

## 22. Gesture Decision Log

The same format as Part 1's [decision log](#6-decision-log): one row per decision, with what was chosen, why, and what else was considered. *(planned)* marks rows about parts that are designed but not built yet ([9.2](#92-what-is-built-and-what-is-not)).

**Decisions from the design**, made before the code was written:

| Decision | Choice | Why | Alternative Considered |
|---|---|---|---|
| Pose model | MoveNet SinglePose Lightning, int8, version 4 ([10.5](#105-movenet)) | The faster MoveNet variant, with a 192 × 192 input and a 2.9 MB file. NXP's eIQ Model Zoo uses the same file and says it "has been tested on i.MX 8MP". Its 17 body keypoints include the wrists, elbows and shoulders that the rules need | Thunder: a 256 × 256 input, "intended for applications that require high accuracy", so more computation per frame. MultiPose: up to 6 people, while one operator is expected ([25.4](#254-adding-gestures-in-the-future)) |
| Recognizing gestures | Hand-written rules on the keypoints ([13.1](#131-why-rules-instead-of-a-second-model)) | No training data exists, and the four gestures are simple geometry. Rules can be explained and unit-tested, and take tens of microseconds per frame | A small learned classifier on keypoint sequences: needs many labelled recordings. Planned for when the list of gestures grows ([25.4](#254-adding-gestures-in-the-future)) |
| Open palm | A stand-in based on the pose: a wrist held still at shoulder or face height, forearm up, for 1 s ([13.7](#137-open-palm)) | MoveNet has no finger keypoints, so it cannot see whether a hand is open. A hand model would add a second model to run, a second NPU graph and code to find the hand: more that can go wrong in a demo | A hand-landmark model run around the wrist, only while a palm is likely ([25.4](#254-adding-gestures-in-the-future)) |
| Language | Python 3.11 | The language of Part 1 and of NXP's Python example for the VX delegate ([19.2](#192-why-litert)). The heavy work (video decoding, resizing, MoveNet) runs in compiled code inside OpenCV and the runtime; the rules, in Python, take about 20 to 30 µs per frame ([19.1](#191-measured-on-the-laptop)) | C++: less overhead, but slower to write, read and change. The measurements so far give no reason for it |
| Frames | One slot that holds only the newest frame ([12.1](#121-keeping-only-the-newest-frame)) | MoveNet always gets the most recent picture. When the loop is slower than the camera, old frames are dropped instead of piling up, so the delay cannot grow | A queue of frames: nothing is dropped, but after any slowdown the service works on the past |
| Frame rate | At most 20 analysed frames per second, `target_fps` ([12.2](#122-the-rate-cap)) | A swipe of 0.3 to 0.6 s still gives 6 to 12 samples. CPU or NPU time is left for FactoryTalk Optix, and the device stays cooler | As many frames as possible: more samples than the rules need, paid for with CPU time and heat |
| MQTT topics | One events topic for all gestures, plus a retained status topic with a last will ([15.1](#151-topics), [15.4](#154-retained-status-and-the-last-will)) | One subscription in FactoryTalk Optix covers every future gesture. The latest status is always available, and the broker announces `offline` when the service dies | One topic per gesture: a new subscription and new HMI configuration for each new gesture |
| Delivery | QoS 1, events not retained, a unique `event_id` in every event ([15.3](#153-qos-1-and-duplicates)) | QoS 1 sends a message again when its receipt is missing, and a copy is recognised by its `event_id`. A retained event would be replayed to FactoryTalk Optix at every restart | QoS 0: a lost message is gone. QoS 2: exactly once, but with a four-step exchange that adds nothing when copies are recognised anyway |
| Repeat protection | In the service: latch, cooldown and return-stroke block ([14](#14-one-gesture-one-event)) | Only the service sees the hand, so only it can tell when a gesture has ended. FactoryTalk Optix adds a second layer that drops copies and old events (planned, [16.2](#162-the-gestureeventmanager-netlogic)) | Only in FactoryTalk Optix: it would have to guess from the timing of the messages |
| HMI side *(planned)* | One global NetLogic, `GestureEventManager`; pages switched by a dynamic link from the PanelLoader's `Panel` to a model variable ([16.2](#162-the-gestureeventmanager-netlogic)) | One place handles every gesture. Every browser session should follow the same variable, so no code per session is needed. To be checked with two browsers | Calling the PanelLoader's `ChangePanel` method from NetLogic, as in Rockwell's sample: the fallback if the link does not switch every session |
| Container user | A normal user, `gesture`, instead of root ([18.1](#181-the-image)) | The service only opens outgoing connections, to the camera and the broker, so it needs no special rights | Root, the user a container gets when the image names none ([Docker run reference](https://docs.docker.com/engine/containers/run/#user)): more rights than the service needs |
| Stuck service | An internal watchdog ends the process; the restart policy `unless-stopped` should then restart the container (not tested) ([17.4](#174-internal-watchdog)) | A restart policy acts when a container exits, not when it is unhealthy ([17.3](#173-heartbeat-file-and-container-healthcheck)). The watchdog turns a hang into an exit | The healthcheck alone: the container would be labelled unhealthy and go on hanging |
| NPU | Tried first when `use_npu` is on; the CPU takes over if the NPU cannot be used ([12.4](#124-running-movenet-and-reading-the-result)) | The demo always runs, and the status reports which one is in use (`accelerator`) | NPU only: a missing or mismatched NPU library would stop the service |
| Quantization on the device | Keep the original file, with its per-channel scales, and measure first ([19.4](#194-how-the-npu-will-be-used)) | A reference project for this chip found that a per-tensor version of MoveNet runs on the NPU, but its keypoint confidence collapses to about 0 | Re-quantizing to per-tensor: possibly faster on the NPU; considered only together with an accuracy check |
| Device network *(planned)* | `network_mode: host` ([11.1](#111-production-on-the-imx-8m-plus)) | The broker is at `127.0.0.1:1883`, as for FactoryTalk Optix, and RTSP video sent over UDP does not pass through address translation (NAT), which can block it. The service opens no listening ports, so sharing the device's network exposes nothing new | A separate container network: more isolation, but the broker gets a different address than the one FactoryTalk Optix uses, and UDP video may not get through the NAT |
| Device camera stream *(planned)* | A low-resolution substream, for example 640 × 360 at about 20 fps ([19.5](#195-bottlenecks)) | The CPU decodes every frame the camera sends, and MoveNet uses only a 192 × 192 picture anyway | The full-resolution main stream: more decoding work for pixels that are thrown away when the picture is shrunk |

**Decisions made or changed while building and testing.** Rows marked "review" fix problems that the two code reviews found ([21.3](#213-test-first-and-the-mutation-check)):

| Decision | Choice | Why | Alternative Considered |
|---|---|---|---|
| Runtime | LiteRT (`ai-edge-litert`), with `tflite_runtime` and full TensorFlow as fallbacks ([19.2](#192-why-litert)) | Measured on the laptop: 3.0 ms (p50), against about 33 ms for `tflite-runtime` 2.14.0, the newest release of the old package on PyPI. A recent NXP Linux release also builds LiteRT | `tflite-runtime`, the plan's first choice and the package in NXP's Python example. It stays in the import list for the device ([12.4](#124-running-movenet-and-reading-the-result)) |
| Clock | `time.perf_counter()` ([12.1](#121-keeping-only-the-newest-frame)) | On Windows with Python older than 3.13, `time.monotonic()` advances in steps of about 16 ms (measured: 0.015625 s), while frames are only 50 ms apart at 20 fps | `time.monotonic()`, which Python 3.13 moved to the same high-resolution clock on Windows |
| Wrist smoothing | None; the rules measure movement over time windows ([13.4](#134-a-track-for-each-wrist)) | Smoothing makes the position lag behind the hand, and the teleport check (`max_wrist_speed`) must see the raw jumps. The time windows already even out jitter (`test_jittery_palm_still_fires_once`) | A running average of each wrist position that weights recent frames more (an exponential moving average), which the plan had |
| Unseen movement (review) | Movement counts only across consecutive frames that saw the wrist. A frame without it, or 0.3 s without any frame (`max_gap_s`), starts a new track ([13.4](#134-a-track-for-each-wrist)) | A jump that was never seen may be MoveNet swapping the wrists. With a time limit alone, 1 to 5 missed frames slipped under it | Requiring a minimum number of samples: still samples before and after the gap fill the count |
| Return-stroke block (review) | The block is measured from the start of the new stroke, and a blocked stroke stays blocked while it keeps moving the same way ([14.3](#143-the-return-stroke)) | A slow return stroke that reaches 1.5 SW only after the block time, or a long one, must not fire | Measuring the block up to the moment the new stroke reaches 1.5 SW: a slow return stroke that began inside the block could fire after it. Downside of the choice: a deliberate opposite swipe that starts within 1.5 s is ignored ([25.2](#252-limitations)) |
| Swipe band (review) | Every sample of a swipe must be inside the chest-to-head band ([13.5](#135-swipe-left-and-swipe-right)) | A pendulum swing has both ends in the band but its middle at hip height | Checking only the ends of the movement |
| Camera errors (review) | Any error while opening or reading is logged and handled like a lost camera ([17.2](#172-camera-loss-and-reconnect)) | Otherwise an unexpected error would end the camera thread silently, and frames would stop arriving | Letting the error end the camera thread |
| NPU acceptance (review) | One test inference right after the NPU delegate loads ([12.4](#124-running-movenet-and-reading-the-result)) | The NPU compiles the network at the first inference. A failure there would end the service instead of falling back to the CPU | Accepting the NPU as soon as the delegate loads |
| Watchdog timing (review) | A deadline: 120 s for startup, then 15 s, renewed on every frame and while the camera is not `running` ([17.4](#174-internal-watchdog)) | Loading the model and the first NPU compilation may take long, and a reconnect loop must survive. A camera that reports `running` but delivers nothing must end the process | One fixed 15 s limit from the start, which a slow NPU startup could exceed |
| Reconnect pause (review) | Back to 1 s whenever frames arrived during the last connection ([17.2](#172-camera-loss-and-reconnect)) | A camera that streamed and then failed is retried after 1 s, not after a doubled pause | Doubling the pause after every failure, whatever happened in between |
| Model delivery | `tools/fetch_model.py` downloads the model before the build, and the image contains it ([18.1](#181-the-image)) | One self-contained image to import into Portainer; nothing to copy onto the device | Mounting the model from the device as a volume: one more file to manage on every device |
| Healthcheck location | In `docker-compose.yml` ([17.3](#173-heartbeat-file-and-container-healthcheck)) | Podman builds OCI images by default. They have no field for a healthcheck, so Buildah drops a `HEALTHCHECK` line | A `HEALTHCHECK` line in the `Dockerfile`: lost in Podman's default image format |
| Webcam address on the laptop | The bridge listens on the WSL adapter address, looked up every time ([18.4](#184-networking-between-windows-podman-and-containers)) | `host.containers.internal` points at the Podman machine, not at Windows. Only this PC can reach the WSL adapter address | Podman's user-mode networking for WSL, `podman machine set --user-mode-networking=true`: not tried. It changes the networking of every running WSL distribution ([podman machine set](https://docs.podman.io/en/latest/markdown/podman-machine-set.1.html)) |
| Laptop container network | `network_mode: bridge`, Podman's default network ([18.4](#184-networking-between-windows-podman-and-containers)) | The network compose created for each project was isolated, so the gesture container could not reach a broker from another compose project | The per-project network that compose creates: connections to the broker timed out |
| Laptop latency target | At most one camera frame interval plus inference ([12.2](#122-the-rate-cap)) | Measured p95: 33 to 35 ms with a 30 fps webcam, mostly a frame waiting while the loop sleeps for the rate cap | The plan's 20 ms, which left out that wait |

## 23. How to Run the Gesture Service

These steps run the gesture chain on a Windows laptop with Podman and the built-in webcam: webcam, bridge, gesture container and broker. A third terminal prints the MQTT messages, so you can watch every gesture arrive. FactoryTalk Optix is not needed, because its part is planned but not built yet ([16](#16-factorytalk-optix-integration-planned)).

Every command is a PowerShell command, run from the repository root: the folder that contains this `README.md`. Do the steps in order. The tests in 23.3 are optional.

When everything runs, you have:

| Where | What runs there | Started in |
|---|---|---|
| Terminal 1 | The webcam bridge | [23.5](#235-start-the-webcam-bridge) |
| In the background, in the Podman machine | The broker container | [23.4](#234-start-an-mqtt-broker) |
| Terminal 2 | The gesture container, showing its log | [23.6](#236-build-and-start-the-container) |
| Terminal 3 | `mosquitto_sub`, printing every gesture message | [23.7](#237-watch-the-messages) |

The diagram of this setup is in [11.2](#112-development-on-a-windows-laptop).

### 23.1 Prerequisites

| What | Used for this guide | Notes |
|---|---|---|
| Windows | Windows 11 Enterprise | |
| Podman | 5.8.1 | A rootful Podman machine with 10 CPUs and 2 GiB of memory ([18](#18-containers-and-the-laptop-setup)) |
| Compose | Docker Compose v5.0.1, started through `podman compose` | Explained below |
| Python | 3.11.9 | Runs the bridge, the model download and the tests |
| Webcam | The laptop's built-in webcam | |
| Internet access | The first time | Downloads the model, the base image and the Python packages |

Other versions were not tried.

**The Podman machine must be running**, because the containers run inside it ([11.2](#112-development-on-a-windows-laptop)). Check:

```powershell
podman machine list
```

`Currently running` in the `LAST UP` column means it runs. Otherwise, start it:

```powershell
podman machine start
```

**What `podman compose` is.** Podman does not read compose files itself. Its documentation describes `podman compose` as "a thin wrapper around an external compose provider such as docker-compose or podman-compose", and adds: "If installed, docker-compose takes precedence" ([podman compose](https://docs.podman.io/en/latest/markdown/podman-compose.1.html)). On the test laptop the provider is Docker Compose, so every `podman compose` command first prints a line like this one:

```text
>>>> Executing external compose provider "…docker-compose.exe". Please see podman-compose(1) for how to disable this message. <<<<
```

The line is harmless. To hide it, set `$env:PODMAN_COMPOSE_WARNING_LOGS = "false"` in the terminal first. The same documentation describes a permanent switch: `compose_warning_logs = false` in the `[engine]` table of Podman's `containers.conf` file.

**A Python virtual environment.** A virtual environment is a folder with its own set of installed Python packages, separate from the main Python installation, so this project's packages cannot clash with those of other projects ([venv](https://docs.python.org/3/library/venv.html)). Create it once, activate it, and install the service's packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r gesture_recognition\requirements.txt
```

- `python -m venv .venv` creates the environment in a new folder, `.venv`. Git ignores that folder (`.gitignore` lists `.venv/`).
- `Activate.ps1` switches this terminal to the environment: the prompt now starts with `(.venv)`, and `python` and `pip` use the environment's packages. This lasts until the terminal is closed, so run the second line again in every new terminal that needs the environment.
- `pip install -r` installs every package listed in the file ([18.1](#181-the-image)). When this guide was written, it installed LiteRT 2.2.0, NumPy 1.26.4, OpenCV 4.11.0.86 (the headless package), paho-mqtt 2.1.0 and PyYAML 6.0.3. The container image installs the same list. TensorFlow is not needed.

### 23.2 Download the model

The model file is not stored in Git ([20.2](#202-the-most-important-lines)). Download it once:

```powershell
python gesture_recognition\tools\fetch_model.py
```

The output, with the start of the path left out:

```text
Downloading MoveNet from https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite ...
Saved 2894840 bytes to …\gesture_recognition\models\movenet_lightning_int8.tflite
```

- Before saving, the script compares the file's SHA-256 fingerprint with the one written in the script ([20.2](#202-the-most-important-lines)). If they differ, it prints `Checksum mismatch: expected …, got …. Nothing saved.` and keeps nothing.
- Running it again is safe. If the file is already there with the right fingerprint, it prints `Model already present: …` and downloads nothing.
- The image build copies this file, so the build stops without it ([18.1](#181-the-image)).

To check the file yourself:

```powershell
(Get-FileHash gesture_recognition\models\movenet_lightning_int8.tflite -Algorithm SHA256).Hash
```

It must print `CD7CC22FA946E5D146A7B98D496853E1923E22828D3972D579973F27F91BB105`. The script writes the same value in small letters; in hexadecimal, `CD` and `cd` are the same digits.

### 23.3 Run the unit tests

This step is optional, but it checks the installation in a few seconds. With the virtual environment active:

```powershell
python -m unittest discover -s gesture_recognition\tests -v
```

`unittest` is Python's built-in test runner ([21](#21-testing-the-gesture-service)). `discover -s` searches the given folder for test files and runs every test in them, and `-v` (verbose) prints one line per test. The run ends like this (the time varies):

```text
Ran 60 tests in 0.661s

OK
```

Two messages in the middle of the run look like errors but are harmless:

- `INFO: Created TensorFlow Lite XNNPACK delegate for CPU.` XNNPACK is a Google library of fast CPU routines. Its README calls it "a highly optimized solution for neural network inference on ARM, x86, WebAssembly, and RISC-V platforms", used by frameworks "such as TensorFlow Lite" ([XNNPACK](https://github.com/google/XNNPACK)). Our code never asks for it: LiteRT creates it by itself. Which of MoveNet's operations it runs was not checked. The same line can also appear in the service log.
- A traceback (Python's error report) that ends with `AttributeError: 'Delegate' object has no attribute '_library'`, printed during `test_missing_npu_delegate_falls_back_to_cpu`, which still reports `ok`. That test asks, on purpose, for an NPU delegate file that does not exist, to check the CPU fallback. Loading fails. Later, when Python throws the half-built delegate object away, LiteRT's clean-up code reads a field, `_library`, that was never set. Python prints that second error under `Exception ignored in: …` and carries on.

Setting `TF_CPP_MIN_LOG_LEVEL` did not hide these lines in our tests.

**Without the model**, the three tests that load the real model file are skipped, each with `skipped 'run tools/fetch_model.py first'`, and the run ends with `OK (skipped=3)`.

**Inside the container.** The image contains no tests ([18.1](#181-the-image)), but the test folder can be mounted into it. Once the image is built ([23.6](#236-build-and-start-the-container)):

```powershell
$root = (Resolve-Path gesture_recognition).Path
podman run --rm -v "${root}\tests:/app/tests:ro" gesture_recognition-gesture:local python -m unittest discover -s tests
```

- `$root` is the full path of the `gesture_recognition` folder.
- Podman's `-v` mounts the test folder at `/app/tests` in the container, read-only (`ro`), and `--rm` deletes the container when the command ends.
- This run does not pass unittest's own `-v`, so it prints one dot per passing test, then the same `Ran 60 tests` and `OK`.

### 23.4 Start an MQTT broker

The service publishes to an MQTT broker ([2.1](#21-what-is-mqtt)). Part 1's stack contains one, the `mqtt-broker` service. Start only that service, in the background:

```powershell
podman compose -f anomaly_detection\docker-compose.yml up -d mqtt-broker
```

- `-f` names the compose file.
- `up` creates and starts containers, and `-d` (detached) runs them in the background and gives the terminal back.
- `mqtt-broker` limits the command to that service, so Part 1's anomaly detector does not start.

The broker publishes port 1883 for MQTT and 9001 for MQTT over WebSocket, and accepts clients without a password ([25.2](#252-limitations)).

> ⚠️ **Not used in our tests on this laptop.** Here, the local image `eclipse-mosquitto:2` is the ARM64 version, which an x86-64 laptop cannot run. Running it printed `WARNING: image platform (linux/arm64/v8) does not match the expected platform (linux/amd64)` and then failed with `Exec format error`. The probable cause is the ARM64 download in Part 1's edge steps ([2B](#2b-deploy-on-edge-device-with-portainer-stack)), which stores the image under the same name.

Check which architecture your local copy has:

```powershell
podman image inspect docker.io/library/eclipse-mosquitto:2 --format "{{.Architecture}}"
```

`amd64` is right for the laptop. If it prints `arm64`, downloading the x86-64 version should replace it. We have not tried this:

```powershell
podman pull --platform linux/amd64 docker.io/library/eclipse-mosquitto:2
```

**Any other broker works**, as long as FactoryTalk Optix uses the same one. Our tests used another Mosquitto broker that was already running on the laptop. If your broker publishes a port other than 1883, give the service that port with `MQTT_PORT` ([23.6](#236-build-and-start-the-container), and "One port, three places" in [18.4](#184-networking-between-windows-podman-and-containers)).

### 23.5 Start the webcam bridge

**Terminal 1.** Open a new PowerShell window in the repository root, and activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Look up the WSL adapter address, and start the bridge on it (rule 4 in [18.4](#184-networking-between-windows-podman-and-containers)):

```powershell
$wsl = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "vEthernet (WSL*").IPAddress
python gesture_recognition\tools\webcam_bridge.py --bind $wsl
```

- `Get-NetIPAddress` lists the computer's network addresses. `-AddressFamily IPv4` keeps the four-number kind, and `-InterfaceAlias "vEthernet (WSL*"` keeps the adapter whose name starts with `vEthernet (WSL`: the `*` stands for any text.
- `( … ).IPAddress` takes only the address, and `$wsl = ` stores it in a PowerShell variable named `wsl`.
- `--bind $wsl` makes the bridge listen on that address.

On the test laptop, the adapter is called `vEthernet (WSL (Hyper-V firewall))` and had the address `172.23.0.1`. The bridge then prints (`…` is the date and time):

```text
… [INFO] Streaming webcam 0 on http://172.23.0.1:8090/stream.mjpg (Ctrl+C to stop)
```

Leave it running. To see the picture, open the printed address in a browser on the laptop. Options such as another webcam or another picture size are in [18.3](#183-why-the-webcam-needs-a-bridge).

### 23.6 Build and start the container

**Terminal 2.** Open another PowerShell window in the repository root. The service runs in the container, so this terminal needs no virtual environment. Look up the address again, because each terminal has its own variables, and tell compose where the camera is:

```powershell
$wsl = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "vEthernet (WSL*").IPAddress
$env:CAMERA_URL = "http://${wsl}:8090/stream.mjpg"
podman compose -f gesture_recognition\docker-compose.yml up --build
```

- `$env:CAMERA_URL = …` sets an environment variable in this terminal. Compose passes it into the container, where it replaces the camera address of `config.yaml` ([18.2](#182-the-compose-file)).
- `${wsl}` needs the braces. PowerShell reads a colon right after a variable name as part of the name, as in `$env:`, so `"http://$wsl:8090/stream.mjpg"` gives `http:///stream.mjpg`, without the address.
- `up --build` builds the image, then starts the container in the foreground, so its log appears in this terminal.

The first build downloads the base image `python:3.11-slim` and the Python packages. Later builds reuse every layer whose input did not change ([18.1](#181-the-image)).

The variables that compose passes to the container:

| Variable | Default in the compose file | Set it when |
|---|---|---|
| `CAMERA_URL` | `http://host.containers.internal:8090/stream.mjpg` | Always on Windows: the default does not reach a bridge that runs on Windows ([18.4](#184-networking-between-windows-podman-and-containers)) |
| `MQTT_HOST` | `host.containers.internal` | The broker is not a container with a published port on this laptop ([18.4](#184-networking-between-windows-podman-and-containers)) |
| `MQTT_PORT` | `1883` | The broker publishes another port, for example `$env:MQTT_PORT = "1884"` before `up` |
| `USE_NPU` | `false`, fixed in the file | Never on the laptop: it has no NPU |

**The log.** Compose puts the container's name in front of every line. After the name, expect lines like these. `…` stands for dates and times, and the first lines can come in another order, because MQTT connects in the background ([15.4](#154-retained-status-and-the-last-will)):

```text
… [INFO] pose: MoveNet 192x192 loaded on CPU
… [INFO] mqtt_client: Connected to MQTT broker host.containers.internal:1883
… [INFO] gesture-recognizer: state=starting fps=0.0 inference p50=0.0 ms processing p95=0.0 ms person=False accelerator=cpu
… [INFO] camera: Camera opened: http://172.23.0.1:8090/stream.mjpg
… [INFO] gesture-recognizer: state=running fps=19.5 inference p50=3.3 ms processing p95=… ms person=True accelerator=cpu
… [INFO] mqtt_client: Gesture swipe_right (confidence 0.87) id=0b269db3-c8ea-4852-ab74-80ecda3c01e5
```

| Line | Meaning |
|---|---|
| `MoveNet 192x192 loaded on CPU` | The model is loaded and runs on the CPU, the only choice on the laptop |
| `Connected to MQTT broker …` | The service reached the broker |
| `state=…` | A summary of the status message: one at the start, then one about every 30 s ([17.3](#173-heartbeat-file-and-container-healthcheck)). `fps` is frames analysed per second, `inference p50` the typical time for image preparation plus MoveNet, `processing p95` the time from frame received to decision, and `person` whether someone passes the presence gate ([13.2](#132-the-presence-gate)) |
| `Camera opened: …` | The service reads the bridge's stream |
| `Gesture … id=…` | A gesture fired, and its event was published ([15.2](#152-payloads)) |

The word between `[INFO]` and the colon names the part of the service that wrote the line: `pose.py`, `mqtt_client.py`, `camera.py`, or `main.py`, which logs as `gesture-recognizer`. The numbers in the example, 19.5 fps and 3.3 ms, are the ones we measured with this setup ([19.1](#191-measured-on-the-laptop)).

**Try the gestures.** Stand facing the webcam, near the middle of the picture, with your shoulders and hands in view. Each gesture gives one `Gesture` line in the log:

| Gesture | How to make it | The rule |
|---|---|---|
| `open_palm` | Raise one hand between chest and face height, forearm up like a "stop" sign, and hold it still for 1 s. Lower the arm to your side before the next one | [13.7](#137-open-palm), [14.1](#141-the-latch-and-its-release) |
| `palm_up` | With the arm hanging at your side, push the hand quickly straight up, above your head, in about half a second | [13.6](#136-palm-up) |
| `swipe_right`, `swipe_left` | With one hand at chest height, sweep it quickly sideways, over at least one and a half shoulder widths: towards your right for `swipe_right`, towards your left for `swipe_left`. Wait a little more than 1.5 s before swiping the other way | [13.5](#135-swipe-left-and-swipe-right), [14.3](#143-the-return-stroke) |

**Run it in the background instead.** Add `-d`, and read the log with `podman logs`:

```powershell
podman compose -f gesture_recognition\docker-compose.yml up --build -d
podman logs -f gesture_recognition-gesture-1
```

`gesture_recognition-gesture-1` is the container's name: compose builds it from the project (the name of the compose file's folder), the service and a number. `-f` (follow) keeps printing new lines. Ctrl+C stops only the printing, not the container.

**The variables stay set** in Terminal 2 until it is closed. To remove one earlier: `Remove-Item Env:CAMERA_URL`.

**Without a container.** The service can also run directly with Windows Python. That skips the image build, which is handy when trying settings ([24](#24-tuning-the-gesture-parameters)). The service then reads the webcam itself, so first stop the bridge (Ctrl+C in Terminal 1), so that it does not hold the webcam. Then, in a terminal with the virtual environment active:

```powershell
$env:CAMERA_URL = "0"
$env:MQTT_HOST = "127.0.0.1"
python gesture_recognition\src\main.py
```

- `0` is the number of the first webcam ([17.2](#172-camera-loss-and-reconnect)).
- `127.0.0.1` reaches the broker's published port from Windows. The comment at the top of `main.py` suggests `localhost`, which also works, but connecting took about 2 s longer in our tests ([18.4](#184-networking-between-windows-podman-and-containers)).
- Ctrl+C stops the service: it logs `Stopping` and publishes `offline` ([15.4](#154-retained-status-and-the-last-will)).
- Never run it while the container runs. Both connect with the client ID `gesture-recognizer`, and the broker disconnects one when the other connects ([15.1](#151-topics)).

> ⚠️ **Partly tested.** Directly on Windows, the service ran with a looping video file, not with `CAMERA_URL=0`. A webcam number is opened with OpenCV's default backend, without the open and read time limits used for streams ([17.2](#172-camera-loss-and-reconnect)).

### 23.7 Watch the messages

**Terminal 3.** The broker image includes Mosquitto's command-line tools. `mosquitto_sub` subscribes to topics and prints every message that arrives. Run it inside the broker container:

```powershell
podman exec anomaly_detection-mqtt-broker-1 mosquitto_sub -h 127.0.0.1 -t "gesture/paperclip/#" -v
```

- `podman exec` runs a command inside a running container. `anomaly_detection-mqtt-broker-1` is the broker container's name, built like the one in 23.6. `podman ps` lists the names of all running containers.
- `-h 127.0.0.1` connects to the broker in the same container, as the stack's own healthcheck does.
- `-t "gesture/paperclip/#"` subscribes to both gesture topics: `#` matches everything below `gesture/paperclip/` ([15.1](#151-topics)).
- `-v` prints each message's topic in front of it.

Example output. The status line is illustrative; the event is a real one, from [15.2](#152-payloads):

```text
gesture/paperclip/status {"v": 1, "state": "running", "person_present": true, "fps": 19.5, "inference_ms": 3.3, "accelerator": "cpu", "timestamp": "…"}
gesture/paperclip/events {"v": 1, "event_id": "0b269db3-c8ea-4852-ab74-80ecda3c01e5", "gesture": "swipe_right", "confidence": 0.87, "timestamp": "2026-10-03T06:25:06.883Z"}
```

- A status line appears at once, because the broker keeps the last status (it is **retained**), and then a new one every 2 s ([15.4](#154-retained-status-and-the-last-will)).
- Each gesture gives exactly one line on `gesture/paperclip/events`.

With another broker, change the container name in the command, or use any MQTT client on Windows, connected to `localhost` and the broker's port.

### 23.8 Stop everything

**The gesture container.** Press Ctrl+C in Terminal 2. Docker's documentation of `compose up`: "If the process is interrupted using SIGINT (ctrl + C) or SIGTERM, the containers are stopped, and the exit code is 0" ([compose up](https://docs.docker.com/reference/cli/docker/compose/up/)). Or, from any terminal in the repository root, stop the container and also remove it:

```powershell
podman compose -f gesture_recognition\docker-compose.yml down
```

Either way, the service receives a stop signal and publishes `offline` before it ends ([15.4](#154-retained-status-and-the-last-will)). We checked this with `down`.

**The rest:**

- The bridge (Terminal 1) and `mosquitto_sub` (Terminal 3): Ctrl+C.
- The broker: `podman compose -f anomaly_detection\docker-compose.yml down`.
- The virtual environment: `deactivate`, or close the terminal.

### 23.9 Troubleshooting

| What you see | Likely cause | What to do |
|---|---|---|
| `Activate.ps1` fails with "running scripts is disabled on this system" | PowerShell's execution policy blocks scripts | If your company's rules allow it, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, as the [venv documentation](https://docs.python.org/3/library/venv.html) shows. Or skip activation, and type `.\.venv\Scripts\python.exe` instead of `python` |
| Typing `$wsl` prints nothing | No adapter matched. Probably WSL is not running | `podman machine start`, then look the address up again |
| `Cannot open webcam 0 (in use by another app?)` | Another program is using the webcam: a video call, for example, or the service started without a container | Close the other program, or pick another webcam with `--device 1` |
| `Camera http://…/stream.mjpg lost; retrying in 1 s`, then 2, 4, 8 and 10 s, and never `Camera opened` | The service cannot reach the bridge. The bridge is not running, the address is wrong, or the WSL address changed | Check Terminal 1. Look up `$wsl` again in both terminals, then restart the bridge and the container |
| `Connection … failed: Connection refused` from FFmpeg, and an OpenCV warning that a backend "is generally available but can't be used to capture by name" | The camera address cannot be reached. If the address in the log contains `host.containers.internal`, `CAMERA_URL` was not set in the terminal that started compose | Set `CAMERA_URL` as in [23.6](#236-build-and-start-the-container) |
| No `Connected to MQTT broker` line, and `MQTT offline: dropped … event` after a gesture | The broker cannot be reached. It is not running, or the host or the port is wrong | Start the broker ([23.4](#234-start-an-mqtt-broker)), and check `MQTT_HOST` and `MQTT_PORT` ([15.5](#155-why-events-are-dropped-while-offline)) |
| `MQTT connection refused: …` | The broker answers but refuses the service, for example because it requires a user name and password | Use a broker without a login. The service has no login settings |
| `MQTT disconnected (…); reconnecting`, again and again | Two programs connect with the same client ID, `gesture-recognizer`, and keep pushing each other off the broker | Run only one copy of the service ([15.1](#151-topics)) |
| `Exec format error` | The image was built for ARM64 | See [23.4](#234-start-an-mqtt-broker) |
| The build stops at `COPY models/ models/` | The model was not downloaded | [23.2](#232-download-the-model) |
| `Checksum mismatch: …` | The download was damaged, or the file on the server changed | Nothing was saved. Try again. If it keeps happening, the file on the server has probably changed |
| `person=False` in the log, `"person_present": false` in the status | You are too far away or off-centre, or your shoulders are hidden | Come closer, to the middle of the picture ([13.2](#132-the-presence-gate)) |
| `No progress before the watchdog deadline; exiting so the container restarts` | The main loop stopped making progress | The service ends itself so that the restart policy can start it again ([17.4](#174-internal-watchdog)). If it keeps happening, read the log lines before it |

Harmless messages, explained above: the compose provider line ([23.1](#231-prerequisites)), and the XNNPACK line and the `'_library'` traceback ([23.3](#233-run-the-unit-tests)).

**If the base image cannot be downloaded.** Some networks block downloads from container registries. If an x86-64 copy of `python:3.11-slim` is already on the laptop, the build can start from it. `podman images` lists the local images with their IDs, and `podman image inspect IMAGE_ID --format "{{.Architecture}}"` shows the architecture of one of them: it must say `amd64`. Then replace the last line of 23.6 with these two, putting the ID in place of `IMAGE_ID`:

```powershell
podman build --from IMAGE_ID -t gesture_recognition-gesture:local gesture_recognition
podman compose -f gesture_recognition\docker-compose.yml up --no-build
```

`--from` "Overrides the first FROM instruction within the Containerfile" ([podman build](https://docs.podman.io/en/latest/markdown/podman-build.1.html)), so the build uses the local image instead of downloading one. `--no-build` makes compose start the image that was just built. The image on our test laptop was built this way.

## 24. Tuning the Gesture Parameters

Every setting of the gesture service is in `gesture_recognition\config.yaml`, with a comment on each line. This section lists them all, says what each one does, and shows how to change them safely. The gesture values are reasoned design values: none of them has been tuned on recordings of real people yet ([25.2](#252-limitations)).

**When a change takes effect.** The service reads the file once, when it starts.

- **In the container**, the file is copied into the image when the image is built ([18.1](#181-the-image)). Rebuild and restart with `podman compose -f gesture_recognition\docker-compose.yml up --build`.
- **Directly on Windows** ([23.6](#236-build-and-start-the-container)), stop the service with Ctrl+C and start it again. No build is needed, so this is the quicker way to try values.

**Four settings can come from environment variables.** These four differ between the laptop and the device, so they are set outside the file: in the compose file ([18.2](#182-the-compose-file)), or in the terminal before starting ([23.6](#236-build-and-start-the-container)). A variable that is set and not empty replaces the value from the file:

| Variable | Replaces | Note |
|---|---|---|
| `CAMERA_URL` | `camera.url` | |
| `MQTT_HOST` | `mqtt.broker_host` | |
| `MQTT_PORT` | `mqtt.broker_port` | Must be a whole number |
| `USE_NPU` | `model.use_npu` | `1`, `true` or `yes`, in small or capital letters, turn the NPU on. Any other value turns it off |

**A misspelt key stops the service.** The code reads every key by its exact name. If one is misspelt, the service stops at startup with a Python error in the log: a `KeyError` for the `camera`, `model` and `mqtt` sections, a `TypeError` for `gestures`.

**Units.** Hand positions are in shoulder widths (SW), measured from the middle of the shoulders: x grows towards the operator's right, and y grows upwards ([13.3](#133-body-coordinates-in-shoulder-widths)). Times are in seconds.

**Camera** (`camera:`)

| Setting | Default | What it does | More |
|---|---|---|---|
| `url` | `http://host.containers.internal:8090/stream.mjpg` | Where the frames come from: a stream address (`http://…` or `rtsp://…`), a webcam number such as `0`, or a video file. Usually replaced by `CAMERA_URL` | [17.2](#172-camera-loss-and-reconnect) |
| `mirror` | `true` | Names directions from the operator's point of view, as in a mirror. With `false`, they follow the camera's view, so `swipe_left` and `swipe_right` swap | [13.3](#133-body-coordinates-in-shoulder-widths) |
| `open_timeout_ms` | `5000` | How long to wait for a stream to open. Streams and files only, not a webcam number | [17.2](#172-camera-loss-and-reconnect) |
| `read_timeout_ms` | `2000` | How long to wait for a frame before the camera counts as lost. Streams and files only | [17.2](#172-camera-loss-and-reconnect) |
| `reconnect_max_s` | `10` | The longest pause between reconnection attempts: 1, 2, 4, 8, then 10 s | [17.2](#172-camera-loss-and-reconnect) |

**Model** (`model:`)

| Setting | Default | What it does | More |
|---|---|---|---|
| `model_path` | `models/movenet_lightning_int8.tflite` | The model file, relative to the `gesture_recognition` folder (`/app` in the container) | [23.2](#232-download-the-model) |
| `use_npu` | `false` | Try the NPU first, and fall back to the CPU if it cannot be used. Usually replaced by `USE_NPU` | [12.4](#124-running-movenet-and-reading-the-result) |
| `vx_delegate_path` | `/usr/lib/libvx_delegate.so` | Where the NPU delegate file is on the device | [19.4](#194-how-the-npu-will-be-used) |
| `num_threads` | `2` | How many CPU cores the model may use, leaving the others for FactoryTalk Optix | [12.4](#124-running-movenet-and-reading-the-result) |
| `target_fps` | `20` | The most frames analysed per second: the rate cap. Setting the camera to about this rate saves decoding work and waiting time | [12.2](#122-the-rate-cap), [19.3](#193-expected-on-the-imx-8m-plus) |

**MQTT** (`mqtt:`)

| Setting | Default | What it does | More |
|---|---|---|---|
| `broker_host` | `host.containers.internal` | The broker's address. Usually replaced by `MQTT_HOST` | [18.4](#184-networking-between-windows-podman-and-containers) |
| `broker_port` | `1883` | The broker's port. It must be the broker that FactoryTalk Optix uses. Usually replaced by `MQTT_PORT` | [18.4](#184-networking-between-windows-podman-and-containers) |
| `client_id` | `gesture-recognizer` | The service's name at the broker. It must be unique | [15.1](#151-topics) |
| `events_topic` | `gesture/paperclip/events` | The topic of the gesture events | [15.1](#151-topics) |
| `status_topic` | `gesture/paperclip/status` | The topic of the status messages | [15.1](#151-topics) |
| `status_interval_s` | `2` | Seconds between two status messages | [17.3](#173-heartbeat-file-and-container-healthcheck) |

Two warnings about the MQTT settings:

- If you rename a topic, FactoryTalk Optix must subscribe to the new name ([16.1](#161-what-the-hmi-will-need)).
- `status_interval_s` does more than its name says. It is also the time window for `fps` and `inference_ms`, and it sets how often the heartbeat file is written, which the healthcheck wants younger than 10 s ([17.3](#173-heartbeat-file-and-container-healthcheck)). The planned HMI check expects a status at least every 6 s ([16.2](#162-the-gestureeventmanager-netlogic)), and the log prints every 15th status. Leave it at 2 s unless you adjust those too.

**Gesture rules: who counts as the operator, and tracking** (`gestures:`)

| Setting | Default | What it does | More |
|---|---|---|---|
| `min_keypoint_score` | `0.3` | Keypoints with a lower score are ignored | [10.4](#104-pose-estimation-and-keypoints), [13.2](#132-the-presence-gate) |
| `min_shoulder_width` | `0.12` | The narrowest shoulder width, as a fraction of the picture. Narrower shoulders mean the person is too far away | [13.2](#132-the-presence-gate) |
| `max_center_offset` | `0.35` | How far, sideways, the middle of the shoulders may be from the middle of the picture. Farther means a passer-by | [13.2](#132-the-presence-gate) |
| `max_wrist_speed` | `15.0` | In SW per second. A faster jump is a tracking glitch, and starts a new track. At 20 fps, that is 0.75 SW from one frame to the next | [13.4](#134-a-track-for-each-wrist) |
| `max_gap_s` | `0.3` | A pause this long between frames, for example a camera hiccup, starts a new track. A frame that misses the wrist always does | [13.4](#134-a-track-for-each-wrist) |
| `lost_s` | `0.5` | A person not seen for this long is forgotten. A hand not seen for this long counts as lowered | [14.1](#141-the-latch-and-its-release), [14.4](#144-forgetting-a-person-or-the-camera) |

**Gesture rules: repeat protection**

| Setting | Default | What it does | More |
|---|---|---|---|
| `cooldown_s` | `1.0` | The shortest time between two gestures | [14.1](#141-the-latch-and-its-release) |
| `opposite_swipe_block_s` | `1.5` | An opposite swipe that starts this soon after a swipe is ignored: it is probably the return stroke | [14.3](#143-the-return-stroke) |
| `release_y` | `-1.0` | `open_palm` and `palm_up` can fire again only after the hand goes below this height | [14.1](#141-the-latch-and-its-release) |
| `max_off_axis_ratio` | `0.5` | Swipes must be mostly sideways and palm-up mostly vertical: the movement in the other direction may be at most half of the main one | [13.5](#135-swipe-left-and-swipe-right), [13.6](#136-palm-up) |

**Gesture rules: the gestures**

| Setting | Default | What it does | More |
|---|---|---|---|
| `palm_hold_s` | `1.0` | How long the hand must be held still for `open_palm` | [13.7](#137-open-palm) |
| `palm_zone_min_y`, `palm_zone_max_y` | `-0.6`, `0.8` | The lowest and the highest height of the held hand | [13.7](#137-open-palm) |
| `palm_still_radius` | `0.2` | How far the held hand may wander from its average position | [13.7](#137-open-palm) |
| `palm_min_forearm_rise` | `0.3` | How far the wrist must be above the elbow | [13.7](#137-open-palm) |
| `up_min_rise` | `1.2` | How far the hand must rise for `palm_up` | [13.6](#136-palm-up) |
| `up_window_s` | `0.7` | The time in which it must rise that far | [13.6](#136-palm-up) |
| `up_end_y` | `1.0` | The height the hand must reach | [13.6](#136-palm-up) |
| `swipe_min_dx` | `1.5` | How far the hand must move sideways for a swipe | [13.5](#135-swipe-left-and-swipe-right) |
| `swipe_window_s` | `0.6` | The time in which it must move that far | [13.5](#135-swipe-left-and-swipe-right) |
| `swipe_band_min_y`, `swipe_band_max_y` | `-1.2`, `1.0` | The height band that the whole stroke must stay in | [13.5](#135-swipe-left-and-swipe-right) |

The heights side by side are in the table at the end of [13.3](#133-body-coordinates-in-shoulder-widths).

**How to tune.** No recordings exist yet; making them is planned work ([25.3](#253-roadmap)). The steps below are the intended method.

**1. Change one value at a time**, and write down what you changed. Otherwise you cannot tell which change helped.

**2. Try values directly on Windows.** Edit `config.yaml`, then restart the service as in [23.6](#236-build-and-start-the-container), "Without a container". While you experiment, the unit test `test_config_yaml_matches_code_defaults` fails; step 4 explains why.

**3. Replay recordings instead of repeating the gesture.** `CAMERA_URL` also accepts a video file. A file plays at its own frame rate, so the time windows see the real timing, and at its end it starts again ([17.2](#172-camera-loss-and-reconnect)). For example, with the virtual environment active:

```powershell
$env:CAMERA_URL = "C:\clips\swipe_right_01.mp4"
$env:MQTT_HOST = "127.0.0.1"
python gesture_recognition\src\main.py
```

`C:\clips\swipe_right_01.mp4` is an example name. Use a full path: a relative one is looked up from the terminal's current folder. For one pass of the file, count the `Gesture …` lines in the log. A recording of one gesture should give exactly one line, and a recording without gestures, such as a person walking past or talking with their hands, should give none. Without a broker, the same events appear as `MQTT offline: dropped … event` lines instead ([15.5](#155-why-events-are-dropped-while-offline)). If the recording app saved a mirrored video, left and right swap: check with a recording of a known swipe. Replaying in the container would need the folder mounted into it, which the compose file does not do yet.

**4. Make a kept value permanent in two places.** The test `test_config_yaml_matches_code_defaults` checks that the `gestures` section of `config.yaml` equals the defaults written in `GestureConfig` in `src/gestures.py`. That way, the unit tests always check the values that are deployed. When you keep a new value, change it in both files, and run the tests ([23.3](#233-run-the-unit-tests)). Tests built around the old value may then fail: decide whether the test or the new value is wrong.

**5. First no false gestures, then sensitivity.** At a trade show, a page that turns by itself is worse than a gesture that must be repeated. So first make the recordings without gestures produce none, then make the real gestures easier to trigger, and check the recordings without gestures again after every change.

**6. Find the distance limit.** `min_shoulder_width` sets how far from the camera the operator may stand, and that distance depends on the camera's lens ([13.2](#132-the-presence-gate)). Watch `person_present` in the status messages ([23.7](#237-watch-the-messages)), which arrive every 2 s, while someone steps slowly back from the camera. Where it turns `false` is the limit. A mark on the floor can show visitors where to stand.

**7. Changing the frame rate.** The rules use seconds, not frame counts, but at very low frame rates no gesture can fire ([19.3](#193-expected-on-the-imx-8m-plus), "At lower frame rates").

## 25. Gesture Status and Next Steps

This section gathers in one place what works, what is missing and what comes next. Each point links to the section that explains it.

### 25.1 Status

In one sentence: the gesture service works on the laptop, and the HMI side and the device come next. The full table is in [9.2](#92-what-is-built-and-what-is-not). In short:

- ✅ **Done and verified:** the service on the laptop, from the webcam through MoveNet and the rules to MQTT. It reports its health and recovers from a lost camera. Its 60 unit tests pass, natively and in the container.
- ⚠️ **Partly verified:** the gestures themselves. The rules are tested with synthetic poses only, and the thresholds are not yet tuned on recordings of real people.
- 📝 **Designed or written, not run:** the FactoryTalk Optix side ([16](#16-factorytalk-optix-integration-planned)), and the NPU code path with its two device tools ([19.4](#194-how-the-npu-will-be-used)).
- ❌ **Not started:** the ARM64 image, the edge compose file and the Ethernet IP camera.

### 25.2 Limitations

These limits are known and accepted for now.

**Camera and model**

- **No fingers.** "Open palm" means a raised, still hand with the forearm up. A fist held the same way also counts ([10.4](#104-pose-estimation-and-keypoints), [13.7](#137-open-palm)). A hand model could fix this ([25.4](#254-adding-gestures-in-the-future)).
- **One person at a time.** MoveNet SinglePose reports one person, preferring people near the middle of the frame. The presence gate ignores people who are far away or off-centre ([13.2](#132-the-presence-gate)). What happens when two people stand close together in the middle has not been tested.
- **Fast swipes may be lost.** If motion blur hides the wrist in the middle of a swipe, the track restarts and the swipe does not fire ([13.4](#134-a-track-for-each-wrist), [14.5](#145-known-gaps)).

**Rules**

- **Synthetic tests only.** Every timing and height in the rules comes from computer-generated poses. For example, the rules assume that 1 SW above the shoulder line is about the top of the head, and nobody has checked this on real people ([13.3](#133-body-coordinates-in-shoulder-widths)).
- **A quick "back" is ignored.** An opposite swipe that starts within 1.5 s of a swipe is treated as the return stroke, and "next, back" can even become "next, next" ([14.3](#143-the-return-stroke), [14.5](#145-known-gaps)). Operators must pause about 1.5 s before reversing.
- **Any sideways arm movement at chest height is a swipe**, for example pointing at something or stretching ([14.5](#145-known-gaps)).

**Messages and safety**

- **A late event can still act once.** After a reconnect, an event can arrive twice or late. The planned NetLogic ignores repeats and events older than 5 s, but an event that is late by less than 5 s still acts once ([15.3](#153-qos-1-and-duplicates)).
- **The broker accepts anyone.** In Part 1's edge stack ([2B](#2b-deploy-on-edge-device-with-portainer-stack)), the broker listens on `0.0.0.0` (every network interface), its ports 1883 and 9001 are published on the device, and `allow_anonymous` is `true`, so no password is needed. Once the HMI side exists, any computer that can reach the broker could send a gesture event that the HMI accepts, for example to turn pages during a demo. There are two possible fixes. One is to make the broker reachable only from the device itself. The other is to require a user name and password, sent over an encrypted **TLS** connection. The service has no login settings yet, so the second fix also needs a code change. The decision is planned for phase 3 ([25.3](#253-roadmap)).

**Health and device**

- **Nothing has run on the i.MX 8M Plus yet.** All device times are estimates ([19.3](#193-expected-on-the-imx-8m-plus)), and the NPU path has never been tried ([19.4](#194-how-the-npu-will-be-used)).
- **A real hang was never observed.** The watchdog was checked with stand-in code (stubs). That the restart policy then restarts the container was not tested ([17.4](#174-internal-watchdog), [17.5](#175-who-notices-what)).
- **A hang while opening the camera is not caught.** If opening hangs while the state is `starting` or `camera_lost`, the watchdog does not fire ([17.4](#174-internal-watchdog)).
- **Two startup limits disagree.** The healthcheck allows 60 s for startup, the watchdog 120 s. Both should be aligned once the startup time on the device is measured ([17.3](#173-heartbeat-file-and-container-healthcheck)).
- **The probe does not look for LiteRT.** Check it by hand ([19.4](#194-how-the-npu-will-be-used)).
- **Two failure cases were not measured:** the last will after a silent network failure, expected after 22.5 s ([15.4](#154-retained-status-and-the-last-will)), and a stopped broker while the whole chain runs ([17.5](#175-who-notices-what)).

**Laptop**

- **The webcam number was not tried.** Directly on Windows, the service ran only with a video file, not with `CAMERA_URL=0` ([23.6](#236-build-and-start-the-container)).
- **Lower frame rates are untested.** All tests feed 20 fps ([19.3](#193-expected-on-the-imx-8m-plus)).
- **The repository's broker did not run on our laptop.** The local `eclipse-mosquitto:2` image was the ARM64 version, so our tests used another broker ([23.4](#234-start-an-mqtt-broker)).
- **No recordings exist yet.** The tuning method of [24](#24-tuning-the-gesture-parameters) is planned, and replaying recordings inside the container would need a folder mounted into it.

### 25.3 Roadmap

The remaining work comes in four phases. Phase 1 builds on the page test of phase 0, and phase 2 on the device test of phase 0.

| Phase | Goal | Work | Done when |
|---|---|---|---|
| 0. Open questions | Settle the two questions the design depends on | **On the device:** run `npu_probe.sh` and `bench_pose.py`, then choose path N1, N2 or N3 ([19.4](#194-how-the-npu-will-be-used)). **In FactoryTalk Optix Studio:** a PanelLoader with two pages, its `Panel` linked to a NodeId variable, and a temporary button that changes the variable. Test it with two browsers open ([16.2](#162-the-gestureeventmanager-netlogic)) | A path is chosen, with measured numbers. Both browsers switch pages, or the `ChangePanel` fallback is chosen |
| 1. HMI on the laptop | The whole chain on the laptop: gesture → MQTT → FactoryTalk Optix → action | The Studio settings ([16.1](#161-what-the-hmi-will-need)) and the `GestureEventManager` NetLogic ([16.2](#162-the-gestureeventmanager-netlogic)). The dialog question of 16.2 is answered here, because it needs the NetLogic. End-to-end tests with two browsers open | Each gesture gives exactly one action in both browsers. A palm held for 10 s opens one dialog. A swipe and its return stroke turn one page. An event sent again with the same `event_id`, and an event whose timestamp is 60 s old, are ignored. A lost camera and a stopped broker show on the HMI |
| 2. On the device | The same on the i.MX 8M Plus, with MoveNet on the NPU | An ARM64 image and an edge compose file with `network_mode: host` ([11.1](#111-production-on-the-imx-8m-plus)), the NPU settings and a volume for the graph cache ([19.4](#194-how-the-npu-will-be-used)). The NPU path chosen in phase 0. The Ethernet camera, with a low-resolution H.264 substream ([19.5](#195-bottlenecks)). Measurements: frame rate, times, CPU and memory use, and temperature during a 60-minute run next to FactoryTalk Optix. End-to-end latency, by filming the operator and the screen together in slow motion (this also includes the camera's own delay) | The status shows `"accelerator": "npu"`. MoveNet fits in the 50 ms between frames, and frame received → HMI reaction stays below about 100 ms ([19.3](#193-expected-on-the-imx-8m-plus)). Or the differences are measured and documented |
| 3. Ready for a trade show | No false gestures in front of visitors | Recordings of each gesture by several people, and of movements that are not gestures: walking past, talking with the hands, two people in the frame. Tuning with these recordings ([24](#24-tuning-the-gesture-parameters)). A booth guide: camera height and angle, a floor mark, lighting, a printed gesture card, the pause before reversing a swipe. The broker access decision ([25.2](#252-limitations)). A folder mount to replay recordings in the container. This README updated with the device numbers | Recordings without gestures give no event, and each gesture recording gives exactly one |

### 25.4 Adding gestures in the future

How to add one more rule-based gesture, step by step, is in [16.4](#164-adding-a-new-gesture-later). The table below lists larger ideas. None of them has been started.

| Idea | What it would do | What it needs |
|---|---|---|
| A real open palm | Run a hand model on a square cut out around the wrist, but only while the palm rule is about to fire, to check that the fingers are spread. One candidate is Google's MediaPipe Hand Landmarker. It contains "a palm detection model and a hand landmarks detection model", and the second one finds "21 hand-knuckle coordinates" ([Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)) | A second model, and code to cut out the hand. The published model is listed as "float 16" (16-bit decimal numbers), while our MoveNet file is int8 ([10.6](#106-quantization)). Whether it, or a converted version, can run on the NPU must be checked first. Running it only now and then would keep the average cost low |
| More rule-based gestures | For example both hands raised, or a swipe downwards: new rules on the same keypoints | The six steps of [16.4](#164-adding-a-new-gesture-later). A new rule must not be true at the same time as an existing one |
| A learned gesture classifier | Replace the hand-written rules with a small neural network trained on recorded keypoint sequences. Examples are a 1D-CNN, a convolutional network that slides along time instead of across a picture, or a GRU, a recurrent network that reads a sequence step by step and keeps a memory of what it has seen | Many labelled recordings of every gesture by different people, even more of movements that are not gestures, and a training step ([10.2](#102-training-vs-inference-and-why-we-never-train), [13.1](#131-why-rules-instead-of-a-second-model)). Worth it when there are many gestures, or gestures too complex for simple geometry |
| Several people | MoveNet MultiPose "detects up to 6 poses" ([10.5](#105-movenet)). A new rule would choose the operator among them, for example the largest and most central person | A different model, to be measured on the NPU like the current one |
| Gestures in the 3D digital twin | Step through the twin's guided tour by gesture. Part 1 already passes the anomaly state to the twin through a small file that a NetLogic writes and the page reads every 2 s ([4](#4-architecture--data-flow)). A tour position could travel the same way | A NetLogic change, and a change to the scene's `ui` module, which contains the tour. Because the page reads the file every 2 s, the twin would react more slowly than a page change |
| Control from the HMI | A third topic, `gesture/paperclip/cmd`, on which FactoryTalk Optix switches gestures on and off, or chooses which gestures each page accepts | Code on both sides. The broker question of [25.2](#252-limitations) applies here too: anyone who can reach the broker could send commands |

---

# Part 3 — Anomaly Detection on the NPU

> 📝 **Status.** Measured on the OptixEdge (i.MX 8M Plus, Yocto scarthgap, kernel 6.12, galcore 6.4.11) and shown on the FactoryTalk Optix panel. The container runs our int8 model on the NPU, and nothing is installed on the device's operating system: the device **lends** its NPU stack to the container, read-only. The ladder log ([34](#34-the-ladder-log-predictions-vs-results)) puts every prediction next to the measured result. [27.7](#277-host-or-container-where-each-moving-part-belongs) explains which of the lent pieces could move into the image, and what we learned trying.

## 26. Goal and Ground Rules

Part 1 runs the autoencoder on the ARM CPU. Part 3 moves the **same detector** onto the i.MX 8M Plus NPU of an OptixEdge and **proves** it on the HMI. Three rules shaped every decision:

1. **Install nothing on the device's operating system.** The NPU is used *from inside the detector's container*. The device contributes only what it already has: the NPU kernel driver and its NPU libraries, lent read-only.
2. **Change one thing at a time.** Each step adds exactly one new unknown, so when something fails we know which layer broke ([34](#34-the-ladder-log-predictions-vs-results)).
3. **Prove, don't claim.** A label saying "NPU" is not evidence. We collect evidence from four independent sources ([30.3](#303-the-evidence-ladder)).

What changed, file by file:

| File | What it does |
|---|---|
| `training/quantize.py` | Makes the int8 twin of the trained model, calibrates its threshold, compares float vs int8 |
| `src/model.py` | `InferenceEngine`: chooses the runtime, loads the NPU delegate, falls back to the CPU, records evidence |
| `src/detector.py` | Converts float ↔ int8 at the model's edges, so scores are always computed in float |
| `src/benchmark.py` | Times the same model on CPU and NPU with the same input |
| `src/npu_evidence.py` | Reads the driver version, the NPU graph-cache files and (optionally) driver counters |
| `src/main.py`, `src/mqtt_client.py` | Status, command and benchmark topics; Last Will |
| `Dockerfile.npu` | The NPU image: Python 3.12 on Debian trixie, our application, and LiteRT as a fallback runtime |
| `tools/npu_entrypoint.sh` | At every start, links the device's NPU libraries into the container and checks that nothing is missing ([27.4](#274-the-entrypoint-borrow-only-the-npu-libraries)) |
| `docker-compose.npu.yml` | Override that gives the container the NPU (device node, read-only libs, cache volume) |
| `tools/npu_check.py` | One-shot proof: "can this container run our model on the NPU?" |
| `FTOptix/.../NpuStatusManager.cs`, `NpuModelBuilder.cs` | HMI side: evidence variables and the Benchmark button ([32](#32-the-factorytalk-optix-proof-panel)) |

## 27. What an NPU Needs

### 27.1 The NPU in one paragraph

[10.7](#107-cpu-npu-runtime-and-delegates) introduced the idea: the CPU is a Swiss-army knife, the NPU a power drill. The drill has two quirks that drive everything in this part:

- **It works in 8-bit integers.** It multiplies small whole numbers very fast. A float32 model must first be translated into integers ([28](#28-quantization-on-our-own-model)).
- **It sits behind a driver.** Our program cannot touch the hardware. It asks a chain of libraries, which ask a kernel driver, which programs the NPU.

### 27.2 The chain, from Python to silicon

One `interpreter.invoke()` in our detector travels through this chain. The boxes say who provides each piece:

```mermaid
flowchart TB
  subgraph IMG["In our image"]
    A["detector: Python 3.12 + numpy + paho-mqtt"]
    L["ai-edge-litert 1.2.0 (fallback runtime)"]
  end
  subgraph LENT["Lent by the device, read-only (from its /usr/lib)"]
    R["tflite_runtime 2.16.2 (Python module)"] --> T["libtensorflow-lite.so.2.16.2 (TFLite runtime + XNNPACK)"]
    T -. "load_delegate()" .-> B["libvx_delegate.so (VX delegate)"]
    B --> T
    B --> V["libtim-vx.so (TIM-VX + OVXLIB)"]
    V -- "OpenVX API (public standard)" --> O["libOpenVX.so.1 (+ OpenVXU)"]
    O --> G["libGAL.so"]
    O --> S["libVSC, libCLC, libArchModelSw, libNNArchPerf, kernel binaries"]
  end
  subgraph KERNEL["The device's kernel (shared by every container)"]
    D["galcore driver, reached through /dev/galcore"]
  end
  A --> R
  G -- "ioctl (private protocol)" --> D --> E["NPU hardware (VIP8000)"]
```

Only the top box is ours. Everything that touches the NPU, from the Python module down, comes from the device. The two labeled arrows are where the pieces speak a *public* and a *private* language; [27.7](#277-host-or-container-where-each-moving-part-belongs) shows why that difference decides what could ever move into the image.

### 27.3 Every module, library and driver

Read the table top-down: each row uses the row below it.

| Piece | What it is | What it does for us | Where it comes from |
|---|---|---|---|
| **Our detector** (`src/`) | Python application | Reads sensor windows from MQTT, calls the model, publishes alerts and evidence | Image |
| **paho-mqtt, PyYAML, scikit-learn** | Python packages | MQTT client; reads `config.yaml`; the MinMaxScaler that maps sensor values to [0, 1] | Image (pip) |
| **numpy** 1.26 | Python package | Holds every tensor we pass in and out. TFLite's Python bridge is compiled against numpy's C interface, so the major version is pinned (`<2.0`) | Image (pip) |
| **`tflite_runtime`** 2.16.2 | Python package: `interpreter.py` (the `Interpreter` and `load_delegate` API) plus `_pywrap_tensorflow_interpreter_wrapper.so`, a compiled bridge | Turns Python calls into calls on the C++ runtime. The bridge is machine code built for CPython **3.12**, so the image's Python must be 3.12 ([27.5](#275-which-tflite-runtime)) | **Device**, lent read-only from `/usr/lib/python3.12/site-packages` |
| **`ai-edge-litert`** 1.2.0 | Google's newer packaging of the same runtime (LiteRT), with its own TFLite copy inside | The fallback: runs the model on the CPU when the device has no `tflite_runtime`, and is runtime B of [27.5](#275-which-tflite-runtime) | Image (pip) |
| **`libtensorflow-lite.so.2.16.2`** | The TensorFlow Lite C++ runtime | Loads the `.tflite` file, plans memory, runs each op. Any op not taken by a delegate runs here, on the CPU | **Device**, lent read-only |
| **XNNPACK, ruy** | CPU math libraries inside TFLite | Hand-tuned ARM NEON kernels and matrix multiplication. They are why our model takes only 10 µs on the CPU ([31](#31-why-the-npu-may-be-slower-here)) | Inside `libtensorflow-lite` |
| **`libvx_delegate.so`** | A TFLite **external delegate**: a plug-in loaded at run time through the C function `tflite_plugin_create_delegate` | Looks at the model's graph, claims the ops the NPU supports, and rewrites them as TIM-VX operations. It reports "5 of 5 nodes" to TFLite ([29.2](#292-catching-the-c-evidence)). It is linked against the device's `libtensorflow-lite.so.2.16.2` | **Device**, lent read-only |
| **`libtim-vx.so`** | TIM-VX ("Tensor Interface Module for OpenVX"), VeriSilicon's C++ library, with **OVXLIB** inside | Translates neural-network operations (fully connected, ReLU, sigmoid…) into an OpenVX graph. The `OVXLIB_VERSION 1.2.14` lines in [29.2](#292-catching-the-c-evidence) come from here | **Device**, lent read-only |
| **`libOpenVX.so.1`**, `libOpenVXU.so` | VeriSilicon's implementation of **OpenVX**, a public standard API (from the Khronos Group) for graphs of image and neural-network operations | `vxVerifyGraph` compiles the whole graph into NPU instructions (the 65 ms warm-up); `vxProcessGraph` runs it (each `invoke`). It also writes and reads the `.nb` graph cache ([30.3](#303-the-evidence-ladder)) | **Device**, lent read-only |
| **`libGAL.so`** | The Graphics Abstraction Layer: the user-space half of the Vivante driver | Allocates memory the NPU can read, fills command buffers, submits them, waits. It speaks to the kernel driver with `ioctl` calls whose data layout is private to one driver release. Its version string is our `GalcoreVersion` ([32.2](#322-the-variables-in-modelnpu)) | **Device**, lent read-only |
| **`libVSC.so`**, `libCLC.so` | VeriSilicon shader compiler and its OpenCL C front end | Compile small programs for the NPU's programmable cores, used for operations the fixed neural-network engine does not do itself | **Device**, lent read-only |
| **`libArchModelSw.so`**, `libNNArchPerf.so` | A software model of the NPU's architecture | Used while compiling the graph to estimate the cost of each layout and choose how to split work | **Device**, lent read-only |
| `libNNVXCBinary*`, `libOvx12VXCBinary*`, `libNNGPUBinary*`, `libOvxGPUVXCBinary*` | Libraries of pre-compiled kernels for this chip | Save compiling common kernels at run time. Not every device has all of them; the entrypoint links whichever exist | **Device**, lent read-only |
| **`galcore`** | The Vivante GPU/NPU **kernel driver** (version 6.4.11 on our device) | Owns the hardware: power, interrupts, memory mapping, scheduling the GPU core (GC7000) and the NPU core (VIP8000) between processes. Its `/sys/kernel/debug/gc/clients` list is our hardware evidence ([30.3](#303-the-evidence-ladder)) | **Device kernel** |
| **`/dev/galcore`** | A device file: the door into the kernel driver | Every user-space request enters the kernel here. A container sees it only if it is passed with `--device` | **Device**, passed in |
| **`&ml_vipsi` in the device tree** | The hardware description the kernel reads at boot | `status = "okay"` switches the NPU block on, so `galcore` finds it. Without it there is no NPU for any program | **Device firmware** |
| **Python 3.12, glibc 2.41, libstdc++** | The base image (Debian *trixie*) | Run everything above, including the lent libraries ([27.6](#276-the-c-runtime-rule)) | Image |

Two packages of NXP's Yocto build supply the lent rows. **`imx-gpu-viv`** is the Vivante user-space driver: OpenVX, GAL, VSC, CLC, the architecture models and the kernel binaries. It is released together with the `galcore` kernel driver, one version for both (6.4.11 here). The machine-learning packages (`tensorflow-lite`, `tensorflow-lite-vx-delegate`, `tim-vx`) sit on top of it. Our device's firmware includes both groups. That is why borrowing works here, and why it is the first thing to check on another device.

### 27.4 The entrypoint: borrow only the NPU libraries

The simple idea, "point the container at the device's whole `/usr/lib`", fails. That folder also contains the device's C library (glibc), C++ library (libstdc++), Python… Our image has its own copies. Two copies of glibc in one process crash in confusing ways.

So `tools/npu_entrypoint.sh` does surgery at every start. The whole `/usr/lib` is mounted at `/host/usr/lib` (read-only: nothing can be changed), but only the NPU libraries are linked into `/opt/npu/lib`, and only that folder is put first on the library search path:

```sh
NPU_HOST_LIBS="libvx_delegate.so* libtim-vx.so* libOpenVX.so* libOpenVXU.so* libGAL.so* libVSC.so* ... libtensorflow-lite.so*"
for pattern in $NPU_HOST_LIBS; do
    for lib in "$HOST_LIB_DIR"/$pattern; do
        ln -sf "$lib" "$NPU_LIB_DIR/$(basename "$lib")"
```

Then it runs `ldd` on the delegate. `ldd` lists every library the delegate needs, directly or through TIM-VX, and where the loader finds it. A line ending in `not found` names exactly what is still missing, *before* Python starts. That turns a cryptic crash into a one-line diagnosis. With `NPU_DEBUG=1` it prints the full list, which shows the split of [27.2](#272-the-chain-from-python-to-silicon) on the real system:

```text
[npu] linked 20 host libraries from /host/usr/lib into /opt/npu/lib
[npu] every dependency of /opt/npu/lib/libvx_delegate.so resolves
	libtensorflow-lite.so.2.16.2 => /opt/npu/lib/libtensorflow-lite.so.2.16.2
	libtim-vx.so => /opt/npu/lib/libtim-vx.so
	libstdc++.so.6 => /lib/aarch64-linux-gnu/libstdc++.so.6
	libgcc_s.so.1 => /lib/aarch64-linux-gnu/libgcc_s.so.1
	libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6
	libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6
	libOpenVX.so.1 => /opt/npu/lib/libOpenVX.so.1
	libVSC.so => /opt/npu/lib/libVSC.so
	libGAL.so => /opt/npu/lib/libGAL.so
	libArchModelSw.so => /opt/npu/lib/libArchModelSw.so
	libNNArchPerf.so => /opt/npu/lib/libNNArchPerf.so
[npu] using the host's tflite_runtime (/host/usr/lib/python3.12/site-packages/tflite_runtime)
[npu] /dev/galcore is present
```

Read the right-hand side. Every NPU library resolves to `/opt/npu/lib`, which means it was lent by the device. The C and C++ runtime (`libc`, `libstdc++`, `libm`, `libgcc_s`) resolves to `/lib/aarch64-linux-gnu`, which means it is the image's own. That is the split of the table above, and the reason the C-runtime rule ([27.6](#276-the-c-runtime-rule)) exists: the device's libraries run on the image's glibc.

Last, it borrows the device's Python runtime package the same way: `/host/usr/lib/python3.12/site-packages/tflite_runtime` is linked into `/opt/npu/lib/python`, which goes first on `PYTHONPATH`. `src/model.py` tries `tflite_runtime` before `ai_edge_litert`, so this borrowed runtime wins whenever it is present ([27.5](#275-which-tflite-runtime)).

### 27.5 Which TFLite runtime

**What we assumed, and what the device said.** The plan assumed TFLite 2.19, from a folder name seen earlier. The read-only probe found `/usr/bin/tensorflow-lite-2.16.2/` instead, and `ldd /usr/lib/libvx_delegate.so` settled it:

```text
libtensorflow-lite.so.2.16.2 => /usr/lib/libtensorflow-lite.so.2.16.2
libtim-vx.so => /usr/lib/libtim-vx.so
libOpenVX.so.1 => /usr/lib/libOpenVX.so.1
...
```

The delegate is not a stand-alone plug-in: it is **linked against the device's own TFLite library**. That gives two ways to pair it with a runtime:

| Option | Runtime in the process | Risk |
|---|---|---|
| **A** (default) | The device's `tflite_runtime` 2.16.2, borrowed read-only. It uses the *same* `libtensorflow-lite.so.2.16.2` as the delegate | None known: this is the pairing NXP builds and tests. Needs the image's Python = the device's Python (3.12) |
| **B** (`HOST_TFLITE_RUNTIME=0`) | The image's `ai-edge-litert` 1.2.0, which contains its own TFLite copy | Two TFLite copies in one process. They meet only through the delegate's C interface, which is stable, so it *can* work, but nobody tests that combination |

We measured both on the device ([34](#34-the-ladder-log-predictions-vs-results)): **both ran the model on the NPU, with outputs identical to the CPU**. We keep A as the default because it is the tested pairing. B stays in the image for two reasons: it is the CPU runtime when the device has no `tflite_runtime`, and it is a live experiment that teaches why ABI pairing matters.

**Why "linked against the same library" matters.** A delegate receives pointers to TFLite's internal structures (tensors, nodes, the context). If the runtime that owns those structures is a different TFLite version, a field may sit at a different offset, and the delegate reads garbage. Same library, same layout, no doubt.

`ai-edge-litert` 1.2.0 was originally picked by date (published one day after TensorFlow 2.19.0), to match the 2.19 we expected. The image's `LITERT_VERSION` build argument selects another one.

### 27.6 The C-runtime rule

The lent libraries were compiled by NXP against the device's glibc. glibc is **backward compatible only**: a program built against an older glibc runs on a newer one, never the reverse. So the image's glibc must be **at least as new as the device's**. Measured with `ldd --version`: the device has glibc **2.39**; `Dockerfile.npu` uses Debian *trixie* (glibc 2.41), while Part 1's image uses *bookworm* (glibc 2.36, too old for the lent libraries).

Newer glibc brought one surprise. glibc 2.41 refuses to load a library marked as needing an *executable stack* (a memory area that is both writable and executable, an old security weakness). The LiteRT wheel carries that mark by accident, and its import failed with `cannot enable executable stack`. The image clears the flag with `patchelf --clear-execstack` when it is built. The library does not actually need it, and the image is safer without it.

### 27.7 Host or container: where each moving part belongs

Today the device lends everything from `tflite_runtime` down. It works, and it is the simplest design that keeps rule 1. But it asks a lot of the device: TensorFlow Lite, the VX delegate, TIM-VX, and a `tflite_runtime` compiled for **exactly** our Python version. On an open Yocto image like ours, they are there. A closed product image, built without the machine-learning packages, would leave the detector on the CPU, and our image's Python version would always be chained to the firmware's.

So where should the line between host and container be? There are three places where pieces meet, and they differ in one thing: **is the language spoken there public or private?**

**Below OpenVX, the conversation is private.** `libGAL` and `galcore` talk through `ioctl` calls: binary structures whose fields can change between driver releases. NXP builds both halves from the same source tree and tests them only as a pair. And the kernel driver **cannot** come from an image: a container has no kernel of its own, it uses the host's. So the user-space driver must follow the kernel driver, which means it must come from the device. When a firmware update changes the driver, the lent libraries change with it, automatically.

**Above OpenVX, the conversation is public.** TIM-VX calls OpenVX functions such as `vxCreateGraph`, `vxVerifyGraph` and `vxProcessGraph`. OpenVX is a published standard. Everything above it (TIM-VX, the delegate, TFLite, the Python module) can travel inside the image.

**Analogy: a wall socket.** The wiring inside the wall and the socket belong to the building (kernel driver and user-space driver: the electrician installs them together). The socket's shape is a standard (OpenVX). The appliance (our image) can bring its own cable and plug, and work in any building with that socket. Today we borrow the building's extension cord as well; it fits, but not every building has one.

| | Borrow everything from `tflite_runtime` down (**current**) | Line at OpenVX (**recommended next step**) | Everything in the image |
|---|---|---|---|
| The device must have | `galcore`, the Vivante driver libs, **plus** TFLite, the VX delegate, TIM-VX and a `tflite_runtime` for our Python | `galcore` and the Vivante driver libs (`imx-gpu-viv`) only | Only `galcore`, but… |
| Works on a closed Yocto image without the ML packages | No: CPU fallback | Yes | In theory |
| Our Python version | Chained to the device's (3.12) | Our choice | Our choice |
| TFLite version | Whatever the firmware ships; follows firmware updates | Pinned in the image, the same on every device | Pinned |
| Breaks when | The firmware drops the ML packages or changes Python | The driver release changes, and the image was compiled for another one | Any driver update: the image's `libGAL` would speak an old `ioctl` dialect to a new kernel driver |
| Build effort | `pip install` only | Cross-compiling TFLite, the delegate and TIM-VX (about 25 min) | Same, plus NXP's licensed driver binaries inside the image |

**The recommendation:** the host provides the kernel driver and the Vivante user-space driver, nothing more; the container carries everything above OpenVX. The last column is a trap: it looks the most independent, but it welds the image to one kernel driver release while making the mismatch invisible until run time.

**What we learned trying it.** We built that image: TFLite 2.16.2, the VX delegate and TIM-VX compiled from NXP's sources at the commits of the device's BSP release, with only the Vivante driver lent. It taught the fine print of "OpenVX is a public standard".

TIM-VX uses OpenVX's standard functions *and* VeriSilicon's neural-network extensions (`vxFullyConnectedLayer`, `vxTensorSelectLayer`…). Extensions are vendor-defined, and each driver release adds some. One header says what a release can do: `VX/vx_khr_compatible.h`, a list of switches like `#define VX_TENSOR_SELECT_VX_SUPPORT 1`. TIM-VX reads the switches and compiles each feature in or out.

The first attempt was compiled against the headers bundled with TIM-VX, from driver **6.4.15**; the device runs **6.4.11**. We started it with `LD_BIND_NOW=1`, which makes the loader look up *every* function at start-up instead of at first use, so a missing one shows at once instead of crashing some day in the middle of a run. It failed immediately:

```text
NPU requested but unusable (OSError: /opt/npu/lib/libtim-vx.so: undefined symbol: vxTensorSelectLayer); running on CPU
```

The detector fell back to the CPU, as designed ([29.1](#291-inferenceengine-try-the-npu-fall-back-remember-why)). Then a diff of two lists showed the whole gap: the 109 OpenVX functions that `libtim-vx.so` imports, against the 2 426 symbols the device's `libOpenVX.so.1` exports (both read with `nm -D`). Exactly two were missing: `vxTensorSelectLayer` and `vxInvalidateHandleVSI`. Comparing the two releases' `vx_khr_compatible.h` showed more: 6.4.15 switches on 15 features that 6.4.11 does not have, a few of which change *how* existing functions are called. Turning off only the two missing functions would have hidden the problem, not solved it.

The fix is headers from the matching release. VeriSilicon's public TIM-VX repository kept, at tag `v1.1.57`, the SDK of driver **6.4.11**. With those headers, and with TIM-VX linked using `-z defs` (every function must be found at link time, so a future mismatch fails on the build machine, not on the device), TIM-VX compiled and linked cleanly. NXP's own driver package (`imx-gpu-viv-6.4.11`) contains the real ARM headers too, but downloading it requires accepting NXP's license agreement.

**Status.** That version is not finished: the full image was not completed and validated on the device, so this part keeps the borrowed design, which is measured end to end. The prototype left three rules for whoever finishes it:

1. **Compile for the driver release you will meet.** One image per driver release, with the release written in the Dockerfile. Check the device's release first (`GalcoreVersion` on the panel shows it).
2. **Find missing functions before the device does.** Link with `-z defs`, and start a check container with `LD_BIND_NOW=1`.
3. **Ask the firmware for the driver, not for TensorFlow.** On a closed Yocto image, the requirement for the NPU is `galcore` with `&ml_vipsi` enabled, plus `imx-gpu-viv`. Everything else is the application's business.

## 28. Quantization on Our Own Model

[10.6](#106-quantization) explained scale and zero point with a ruler. Here is the ruler of **our** model.

### 28.1 From float to int8, without retraining

`training/quantize.py` reads the weights out of `models/autoencoder.tflite`, rebuilds the identical Keras network (largest difference 2.98 × 10⁻⁷, a rounding error) and converts it to full int8. It changes only one thing, how the numbers are stored. So any difference between the two files is caused by quantization alone.

The converter needs a **representative dataset**: real inputs, to measure the range of every intermediate value. We pass the scaled training windows, exactly what the model sees in production. Ranges measured on unrealistic data would put the ruler's marks in the wrong place.

### 28.2 The ruler of our input

Measured with the interpreter:

```text
input  int8  scale=0.003922  zero_point=-128  covers [0.000, 1.000]
output int8  scale=0.003906  zero_point=-128  covers [0.000, 0.996]
weights (64, 10)  int8  per-tensor
weights (16, 64)  int8  per-tensor
weights (64, 16)  int8  per-tensor
weights (10, 64)  int8  per-tensor
```

The scale is 0.003922 = 1/255, and the zero point −128 sits at 0.0. So the 256 marks cover exactly [0, 1], the output range of the MinMaxScaler. The converter found this by itself from the representative data.

Worked examples, with `q = round(x / scale) + zero_point` and `back = (q − zero_point) × scale`:

| Real value x | Stored q | Read back | Error |
|---|---|---|---|
| 0.0 | −128 | 0.000000 | 0 |
| 0.25 | −64 | 0.250980 | 0.00098 |
| 0.5 | −1 | 0.498039 | 0.00196 |
| 0.731 | 58 | 0.729412 | 0.00159 |
| 1.0 | 127 | 1.000000 | 0 |
| 1.2 | 127 (clipped) | 1.000000 | 0.2 |

The error is at most half a mark (0.00196). The last row matters: a sensor value **outside** the training range saturates at the edge of the ruler. The float model would have seen 1.2; the int8 model sees 1.0. Detection still works, because the score is computed against the original float input (below), so a saturated input still differs from the reconstruction.

### 28.3 Where the conversion happens

The int8 model takes and returns int8 numbers. `src/detector.py` translates at the edges, using the scale and zero point stored in the model file:

```python
q = np.clip(np.round(x / scale) + zero_point, -128, 127)   # float -> int8, before invoke()
y = (q_out.astype(np.float32) - zero_point) * scale         # int8 -> float, after
score = mean((x - y) ** 2)                                  # always in float, for both models
```

The same code serves both files. For the float model there is no scale, and the translation is skipped.

### 28.4 Per-tensor, and a separate threshold

TFLite quantizes `Dense` weights **per-channel** by default (one ruler per output neuron). Reports for this chip say the NPU prefers **per-tensor** (one ruler per matrix) ([19.4](#194-how-the-npu-will-be-used), Step 3). `quantize.py` asks the converter for per-tensor through a flag whose name begins with `_experimental`, so a future TensorFlow may change it. For MoveNet, per-tensor destroyed accuracy. So we measured instead of assuming:

| | float32 | int8 per-tensor | int8 per-channel |
|---|---|---|---|
| File size | 16 544 B | 7 064 B | 10 648 B |
| Threshold | 0.010150 | 0.010236 | (own) |
| Faults caught (100 sustained-fault windows) | 100 | 100 | 100 |
| False alarms (200 normal windows) | 10 | 10 | 10 |
| Decisions different from float32 | — | **0** | **0** |

Why its own threshold? Rounding adds a small error to every reconstruction, so "normal" scores are slightly larger (0.010236 vs 0.010150). The float threshold would raise false alarms slowly over time. `quantize.py` therefore recalibrates the three-sigma threshold ([3.2](#32-the-math)) on the int8 model's own scores and writes `models/threshold_int8.json`. `config.yaml` keeps each model with its threshold:

```yaml
variants:
  float32: { model_path: models/autoencoder.tflite,      threshold_path: models/threshold.json }
  int8:    { model_path: models/autoencoder_int8.tflite, threshold_path: models/threshold_int8.json }
precision: auto   # int8 when use_npu is true, float32 otherwise
```

The gate passed with zero decision changes, so per-tensor is the default.

## 29. The NPU-Aware Detector

### 29.1 `InferenceEngine`: try the NPU, fall back, remember why

`src/model.py` wraps one TFLite interpreter plus the facts that prove where it runs:

```python
if use_npu:
    try:
        delegate = load_delegate(vx_delegate_path)
        self.interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate])
        self.interpreter.allocate_tensors()
        self.warmup_ms = self._timed_zero_run()   # first invoke: the NPU compiles the graph here
        self.accelerator = "npu"
    except Exception as exc:
        self.npu_error = f"{type(exc).__name__}: {exc}"
        self.interpreter = None
if self.interpreter is None:
    self.interpreter = Interpreter(model_path=model_path)   # CPU, same model file
```

Three details, each for a reason:

- **The warm-up `invoke()` is inside the `try`.** The NPU compiles the graph at the first inference, and a mismatched driver fails exactly there. Outside the `try`, that failure would crash the detector instead of falling back.
- **`accelerator` becomes `"npu"` only after a successful inference.** Loading the delegate is not enough proof.
- **The fallback uses the same int8 file.** The threshold therefore still matches the model.

The runtime is chosen in the order `tflite_runtime` → `ai_edge_litert` → `tensorflow`: whichever imports first. In the NPU container that is the device's `tflite_runtime` 2.16.2, borrowed by the entrypoint, because it shares its TFLite library with the delegate ([27.5](#275-which-tflite-runtime)). With `HOST_TFLITE_RUNTIME=0`, or on a laptop, LiteRT runs the same model.

### 29.2 Catching the C++ evidence

When a delegate takes over part of a graph, many TFLite builds print a line like:

```text
INFO: Replacing 5 out of 5 node(s) with delegate (Vx Delegate) node, yielding 1 partitions.
```

That is evidence, but it is printed by C++ code directly to a file descriptor, below Python's `sys.stdout`/`sys.stderr`. `capture_native_output()` temporarily redirects descriptors 1 and 2 into a file, keeps the lines, and echoes them to the real log. `delegation_summary()` turns them into `{"delegate": "Vx Delegate", "nodes": 5, "of": 5}`. Our graph has 5 nodes: 4 `FULLY_CONNECTED` and 1 `LOGISTIC` (the sigmoid). "5 out of 5" would mean the whole model runs on the NPU; "4 out of 5" would mean the sigmoid stays on the CPU, and data travels CPU → NPU → CPU every inference.

**Two surprises from the device**, both found by measuring:

1. **The device's TFLite 2.16.2 does not print the "Replacing" line at all.** So the code also reads the graph itself (`graph_delegation()`). After a delegate is applied, the interpreter lists the original ops *plus* one `DELEGATE` node per partition:

   ```text
   ['FULLY_CONNECTED', 'FULLY_CONNECTED', 'FULLY_CONNECTED', 'FULLY_CONNECTED', 'LOGISTIC', 'DELEGATE']
   ```

   An original op belongs to the partition if it lies on a path from the `DELEGATE` node's inputs to its outputs. On the device: `Vx Delegate: 5/5 nodes`.
2. **The VX delegate writes its own lines (`INFO: Vx delegate: ...`) to stdout, not stderr.** We found it by running the check twice, once with `2>/dev/null` and once with `>/dev/null`. That is why the capture covers both descriptors, and calls C's `fflush()` before restoring them: when stdout is not a terminal, C keeps it in a buffer that would otherwise be written after the capture ends.

`TF_CPP_MIN_LOG_LEVEL=3` in Part 1's compose file hides INFO lines. The NPU override sets it to `0`.

**Going one level deeper (lab only).** The NPU's own graph compiler, OVXLIB, can describe what it built. With `VSI_NN_LOG_LEVEL=4` on the device it prints every NPU node:

```text
I [vsi_nn_CreateGraph:1327]OVXLIB_VERSION==1.2.14
D [setup_node:535]Setup node id[8] uid[30000] op[DATACONVERT]
D [print_tensor:166]in(0) : ... fmt[i8 ] qnt[ASM zp=-128, scale=0.003922]
D [print_tensor:166]out(0): ... fmt[u8 ] qnt[ASM zp=  0, scale=0.003922]
D [setup_node:535]Setup node id[0] uid[1] op[FCL2]
D [setup_node:535]Setup node id[1] uid[2] op[RELU]
...
D [setup_node:535]Setup node id[7] uid[8] op[SIGMOID]
D [setup_node:535]Setup node id[9] uid[30001] op[DATACONVERT]
```

Read it like a recipe. Our 5 TFLite ops became 4 `FCL2` (fully connected), 3 `RELU` (TFLite had fused the ReLU into each FULLY_CONNECTED; the NPU wants it as a separate step) and 1 `SIGMOID`. And two `DATACONVERT` nodes appear at the edges: this NPU prefers **uint8** (0…255) over int8 (−128…127). Same ruler, shifted: int8 with zero point −128 and uint8 with zero point 0 describe exactly the same real numbers, so the conversion just adds 128 to every byte. The weights were converted the same way (`fmt[u8] zp=128`).

## 30. Evidence over MQTT

### 30.1 Topics

| Topic | Direction | Retained | Content |
|---|---|---|---|
| `anomaly/paperclip/alerts` | detector → HMI | no | As in Part 1, plus `accelerator`, `precision`, `inference_ms` for every window |
| `anomaly/paperclip/status` | detector → HMI | **yes** | Where the model runs and why: accelerator, NPU error, runtime, delegate, delegation, warm-up, driver version, graph-cache files |
| `anomaly/paperclip/cmd` | HMI → detector | no | `{"cmd": "benchmark", "runs": 1000}` (capped at 5000) |
| `anomaly/paperclip/benchmark` | detector → HMI | no | CPU vs active engine: p50, p95, largest output difference |

The full payloads are documented in the `src/mqtt_client.py` docstring. The alert payload only **adds** fields, so the existing `AnomalyFeedbackManager`, which reads `is_anomaly`, keeps working.

### 30.2 Retained status and the Last Will

The status is **retained**: the broker keeps the last copy and gives it to anyone who subscribes later. FactoryTalk Optix can start after the detector and still see the badge at once. The same idea as the gesture service ([15.4](#154-retained-status-and-the-last-will)).

A retained "online, NPU" would lie forever if the detector crashed. So the detector registers a **Last Will** at connect time: `{"state": "offline"}`, retained, same topic. If the connection dies without a goodbye, the broker publishes the will itself, and the HMI badge turns to *offline*.

### 30.3 The evidence ladder

Each rung is harder to fake than the one below. A strong proof uses several.

| Rung | Evidence | Who produces it | Fooled by |
|---|---|---|---|
| 1 | `"accelerator": "npu"` | Our code | A bug in our code |
| 2 | `Vx Delegate: 5/5 nodes`: TFLite's "Replacing N out of M" line, or the `DELEGATE` node in the interpreter's own graph | TFLite itself | Nothing in our code, but it says only that the delegate *accepted* the nodes |
| 3 | A `*.nb` file in the graph cache | The NPU driver stack | Nothing: OpenVX writes it only after compiling the graph *for the NPU*. Measured: 54 976 bytes, and the next start's warm-up drops from about 65 ms to 16 ms because the compilation is skipped |
| 4 | The kernel driver's own bookkeeping lists our process as an NPU client, with memory on the NPU core | The kernel driver | Nothing. Measured: `/sys/kernel/debug/gc/clients` lists `python`; `/proc/<pid>/cgroup` puts it in `docker-9c2e699aadaa…`, the detector container; `/proc/<pid>/fd` shows `/dev/galcore` open; and `/sys/kernel/debug/gc/database` lists that process with `VideoMemory` records on **GPU 1**, which `gc/info` names as the VIP8000 NPU (GPU 0 is the GC7000 graphics core). Root-only, so a host lab check from an SSH shell |
| 5 | **Negative control:** with `USE_NPU=false` rungs 1–4 disappear | The experiment | — Measured: status `cpu` / `float32`, no delegate, benchmark `NPU not requested`, and `gc/clients` **empty** |

**Why rung 4 is only a lab check.** `/sys/kernel/debug` is readable only by the host's root. Our rootless container runs as uid 985, so it gets *Permission denied*. Opening debugfs to the container would need `--privileged` or a rootful daemon, and we refuse both. So the panel shows rungs 1–3, and rung 4 is checked from an SSH shell.

**A counter that lied (to us).** We planned rung 4 as "the NPU's **On** time grows during a benchmark", from `/sys/kernel/debug/gc/idle`. We measured it three times around the same 5 000-run benchmark: it grew by **601 ms** once, and by **0 ms** twice, even when read again 1 s later. The file has one set of numbers for the whole GPU+NPU block, not one per core, and the driver seems to update it only at some power-state changes (`poweroff_timeout` is 300 ms). The lesson: **evidence you cannot explain is not evidence.** We replaced it with the process list above, which says *who* uses the NPU, without timing problems.

The graph cache is enabled by NXP's variables `VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1` and `VIV_VX_CACHE_BINARY_GRAPH_DIR` ([19.4](#194-how-the-npu-will-be-used)). `docker-compose.npu.yml` points the folder to a named volume, so the file survives restarts and later starts skip the compilation.

The **negative control** is what makes the badge trustworthy. A panel that shows "NPU" whatever happens proves nothing. A panel that changes to "CPU", loses the delegation line and the `.nb` file when we switch the NPU off, proves it is reading reality.

## 31. Why the NPU May Be Slower Here

**Prediction: for our model, the NPU will be slower than the CPU. Result: confirmed, 15 times slower per inference.** This is not a failure; it is the most important lesson of this part.

Count the work. Our network is 10 → 64 → 16 → 64 → 10:

```
10×64 + 64×16 + 16×64 + 64×10 = 640 + 1024 + 1024 + 640 = 3 328 multiply-adds per inference
```

The NPU is rated at 2.3 TOPS: it could do our math thousands of times in a microsecond. But each inference also costs a fixed **trip**: copy the input into memory the NPU can read, ask the driver through `/dev/galcore`, wait for the NPU, copy the output back. That trip costs a fixed amount of time, however small the math.

**Analogy: a moving truck for one box.** Moving one box across town, you carry it in your hands (CPU). Hiring a truck (NPU) is faster *per box* but costs an hour of booking and loading. For one box the hands win; for a house the truck wins.

The laptop gave the scale: one int8 inference on the CPU takes **0.029 ms**. We predicted 0.05–0.2 ms on the Cortex-A53 and 0.2–1 ms for one trip to the NPU. Measured on the OptixEdge (1 000 runs, same input, only `invoke()` timed, `tflite_runtime` 2.16.2):

| Model | CPU p50 | NPU p50 | NPU first run (compiles the graph) |
|---|---|---|---|
| int8 | **0.0106 ms** | **0.157 ms** (15× slower) | 72 ms |
| float32 | 0.0129 ms | 0.248 ms | 67 ms |

Both predictions were too pessimistic for the CPU: the A53 with XNNPACK does our 3 328 multiply-adds in about 10 µs. The NPU trip, about 0.15 ms, landed inside the predicted range. **The truck for one box, measured.**

Now the detector's real path, `engine.reconstruct()`, which also converts float → int8, copies the input in, copies the output out and converts back (200 runs, inside the container):

| Path | CPU p50 | NPU p50 | NPU ÷ CPU |
|---|---|---|---|
| `reconstruct()`, `npu_check.py` (200 runs) | 0.347 ms | 0.766 ms | 2.2× |
| `reconstruct()`, the running detector's benchmark from MQTT (5 000 runs) | 0.347 ms | 0.870 ms | 2.5× |
| The same, a later run after a restart | 0.350 ms | 0.656 ms | 1.9× |

The rows differ because the running detector shares the CPU with its MQTT thread, and because the NPU trip varies between runs (0.656 to 0.870 ms for the same benchmark): a single number would hide that spread, which is why the panel shows p95 too. In both cases the largest output difference between CPU and NPU was **0.0**: the same integers came out of both engines.

The alerts report one inference each: 1.3–1.9 ms on the NPU, more than the benchmark's p50. A single call after a pause starts with cold CPU caches and may find the NPU powered down (`poweroff_timeout`: 300 ms). It is still far below the 30 s between windows.

A second lesson hides here: **the Python around the model costs 30 times more than the model itself** on the CPU (0.347 vs 0.0106 ms). Whoever optimizes the "AI" part first is optimizing the wrong thing. Either way, one inference every 30 s costs less than a millisecond: both engines are far more than fast enough.

And a surprise: the delegate also accepted the **float32** model, which we expected it to reject. The VIP8000 can compute in 16-bit floating point, and the driver converts on the fly. It is slower than int8 (0.248 vs 0.157 ms), and its outputs differ slightly from the CPU's (0.5045132 vs 0.50451314: float16 rounding). Quantization is still the right choice for an NPU, but "float does not run at all" was wrong.

So why use the NPU at all? Three reasons:

1. **Proof of the pipeline.** Quantization, delegate, container and evidence are the same for a model 1 000× larger, like MoveNet ([19](#19-performance-and-the-npu-plan)), where the NPU wins clearly.
2. **CPU left for the HMI.** On the OptixEdge, FactoryTalk Optix shares the CPU. Work moved to the NPU is work the HMI does not wait for.
3. **Room to grow.** A bigger model (more tags, longer windows, the V2 model in the root `plan.md`) moves toward the truck's side of the trade-off.

## 32. The FactoryTalk Optix Proof Panel

### 32.1 What the NetLogic does

- **`NpuModelBuilder.cs`** (design-time) creates the folder `Model/NPU` with 35 typed variables, once. It never overwrites: running it twice is safe.
- **`NpuStatusManager.cs`** (runtime) listens to three variables written by MQTT subscriptions, parses the JSON and fills the `Model/NPU` variables the UI binds to. It also exports `RequestBenchmark()`, which publishes `{"cmd":"benchmark","runs":N}` through `MQTTClient.Publish` (QoS 1, never retained, or the detector would re-run the benchmark at every reconnect).

If the status says `offline` (the Last Will), it clears `IsOnNpu`, so a stale NPU badge cannot survive.

### 32.2 The variables in `Model/NPU`

**Why two steps, text then typed variables?** An MQTT subscription can write a whole message into one variable, but only as one string. A badge color needs a Boolean, a bar needs a number. So the subscriptions write the raw JSON into `StatusJson` and `BenchmarkJson`, and `NpuStatusManager` reacts to every change of those strings, reads the fields and writes them into typed variables. The panel binds only to the typed ones. A third input is Part 1's `Model/anomaly_detector_feedback`, which already receives every alert.

```mermaid
flowchart LR
  S["anomaly/paperclip/status"] --> SJ["StatusJson"]
  B["anomaly/paperclip/benchmark"] --> BJ["BenchmarkJson"]
  A["anomaly/paperclip/alerts"] --> AF["Model/anomaly_detector_feedback"]
  SJ --> M["NpuStatusManager (parses JSON)"]
  BJ --> M
  AF --> M
  M --> V["typed variables in Model/NPU"] --> UI["panel widgets"]
  UI -- "Benchmark button: RequestBenchmark()" --> C["anomaly/paperclip/cmd"]
```

The tables give each variable, the JSON field it comes from, and the value on our device with the NPU running.

**Inputs and settings**

| Variable | Type | Written by | Meaning |
|---|---|---|---|
| `StatusJson` | String | MQTT subscription to `anomaly/paperclip/status` | The raw retained status message. Starts as `waiting for detector status ...` |
| `BenchmarkJson` | String | MQTT subscription to `anomaly/paperclip/benchmark` | The raw result of the last benchmark |
| `MqttClientPath` | String | Design time (`MQTT/MQTTClient1`) | Which MQTT client node `RequestBenchmark()` publishes with |
| `CommandTopic` | String | Design time (`anomaly/paperclip/cmd`) | Where the benchmark request is published |
| `BenchmarkRuns` | Int32 | The operator (default 1000) | How many inferences per engine. `RequestBenchmark()` clamps it to 1–5 000, the same cap the detector applies |

**Where the model runs** (from `anomaly/paperclip/status`, refreshed at every detector start and after each benchmark)

| Variable | Type | JSON field | Meaning | On our device |
|---|---|---|---|---|
| `DetectorState` | String | `state` | `online` while the detector is connected; `offline` comes from the Last Will ([30.2](#302-retained-status-and-the-last-will)) | `online` |
| `Accelerator` | String | `accelerator` | The engine that **actually** ran the warm-up inference: `npu` or `cpu`; set to `offline` when the detector is gone | `npu` |
| `IsOnNpu` | Boolean | `accelerator == "npu"` | The badge's color. Forced to false when offline | true |
| `NpuRequested` | Boolean | `npu_requested` | Whether `USE_NPU` asked for the NPU. With `Accelerator`, separates "NPU failed" from "NPU not asked" | true |
| `NpuError` | String | `npu_error` | Empty when the NPU works. Otherwise the exception text, naming the failing layer ([33.6](#336-troubleshooting)) | empty |
| `ModelName` | String | `model` | The model file in use | `autoencoder_int8.tflite` |
| `Precision` | String | `precision` | `int8` or `float32` ([28.4](#284-per-tensor-and-a-separate-threshold)) | `int8` |
| `Runtime` | String | `runtime` | The TFLite Python package that was imported, and its version ([29.1](#291-inferenceengine-try-the-npu-fall-back-remember-why)) | `tflite_runtime 2.16.2` |
| `Delegate` | String | `delegate` | Path of the delegate library loaded inside the container; `none (CPU)` when none | `/opt/npu/lib/libvx_delegate.so` |
| `Delegation` | String | `delegation` | How many of the model's nodes each delegate took ([29.2](#292-catching-the-c-evidence)); `no delegate took any node` on the CPU | `Vx Delegate: 5/5 nodes` |
| `WarmupMs` | Double | `warmup_ms` | Duration of the first inference, where the NPU compiles the graph | about 66 ms; 16 ms with the graph cache |
| `Threshold` | Double | `threshold` | The anomaly threshold of the loaded model variant | 0.010236 |
| `GalcoreVersion` | String | `galcore_version` | The NPU driver's version. The kernel driver's own version file is in sysfs on some devices; on ours it is only in debugfs, readable by root. So the detector falls back to the version string compiled into the lent `libGAL.so` (the user-space half of the same driver release). The prefix says which source was read | `libGAL 6.4.11:745085` |
| `GraphCacheFiles` | Int32 | `graph_cache.files` | Number of compiled NPU graphs (`*.nb`) in the cache volume ([30.3](#303-the-evidence-ladder)) | 1 |
| `GraphCacheBytes` | Int32 | `graph_cache.bytes` | Their total size | 54 976 |
| `GraphCacheNames` | String | `graph_cache.names` | Their file names; `cache disabled` when the cache variables are not set | one `<hash>.nb` |
| `StartedAt` | String | `started_at` | When the detector process started (UTC) | an ISO time stamp |

**Every window** (from the alerts, about every 30 s)

| Variable | Type | JSON field | Meaning |
|---|---|---|---|
| `LastInferenceMs` | Double | `inference_ms` | Time of the last real inference: 1.3–1.9 ms on the NPU, more than the benchmark's p50 because the caches are cold after 30 s ([31](#31-why-the-npu-may-be-slower-here)) |
| `LastAlertAccelerator` | String | `accelerator` | The engine that computed that window. It proves the NPU is used for *production* windows, not only for the warm-up |

**Benchmark** (from `anomaly/paperclip/benchmark`, after the button)

| Variable | Type | JSON field | Meaning | Measured (5 000 runs) |
|---|---|---|---|---|
| `BenchmarkRunning` | Boolean | — | Set true by `RequestBenchmark()`, false when a result arrives. A second click while true is refused | |
| `BenchRuns` | Int32 | `runs` | Inferences per engine | 5 000 |
| `BenchCpuP50Ms`, `BenchCpuP95Ms` | Double | `cpu.p50_ms`, `cpu.p95_ms` | Median and 95th-percentile time of `reconstruct()` on the CPU | 0.350, 0.547 ms |
| `BenchNpuP50Ms`, `BenchNpuP95Ms` | Double | `npu.p50_ms`, `npu.p95_ms` | The same on the active engine; 0 when the NPU is not in use | 0.656, 0.754 ms |
| `BenchNpuVsCpu` | Double | `npu_vs_cpu_p50` | NPU p50 ÷ CPU p50. Above 1 means the NPU is slower ([31](#31-why-the-npu-may-be-slower-here)) | 1.87 (2.5 in an earlier run) |
| `BenchMaxOutputDiff` | Double | `max_output_difference` | Largest difference between the CPU and NPU outputs for the same input | 0.0 |
| `BenchMessage` | String | built by the NetLogic | A sentence for the operator: the ratio explained, `No NPU run: <reason>` (from `npu_unavailable`), or an error | |
| `BenchFinishedAt` | String | `finished_at` | When the benchmark ended | |
| `NpuCounterDelta` | String | `counters_before`, `counters_after` | Change of driver counters during the benchmark. Here always `n/a (driver counters not exposed to the container)`: the rootless container cannot read debugfs ([30.3](#303-the-evidence-ladder)) | n/a |

Percentiles, briefly: run the same inference 5 000 times and sort the times. p50 is the one in the middle (the typical case), p95 the one at position 4 750 (a bad but common case). A mean would be dragged up by a few slow runs, when the operating system paused our process.

### 32.3 Reading the panel

| You see | It means |
|---|---|
| Badge NPU, delegation `Vx Delegate: 5/5`, a `.nb` file | The whole model runs on the NPU, proven by three independent sources |
| Badge NPU, delegation `4/5` | The sigmoid stays on the CPU; each inference crosses CPU ↔ NPU twice |
| Badge CPU, `NpuError` filled | The NPU was requested but failed; the text says at which layer ([33.6](#336-troubleshooting)) |
| Badge CPU, `NpuError` empty, precision `float32` | The NPU was not requested (`USE_NPU=false`): the negative control |
| `BenchNpuVsCpu` > 1 | The NPU is slower for this model: expected ([31](#31-why-the-npu-may-be-slower-here)) |
| `BenchMaxOutputDiff` about 0.004 or less | CPU and NPU compute the same int8 model; differences are at most about one quantization mark. Measured: 0.0 |
| Badge `?` | No status message has arrived: check the subscription to `anomaly/paperclip/status` and the broker port |

## 33. Deploying to the Device

> ⚠️ The OptixEdge is a production-style device. A deployment creates only Docker objects named `anomaly_detection-*`, which can be removed. Never use `--privileged`, never install packages on the host, never `docker system prune`.

### 33.1 What the container must receive

Whatever tool deploys it, the detector container needs exactly these things from the device, and nothing more:

| What | In the compose file | Why |
|---|---|---|
| The image `localhost/anomaly_detection-detector:npu` (arm64) | `image:` with `pull_policy: never` | It is not in a public registry, so it is imported as a file. `never` stops Docker from looking for it online |
| The NPU's door | `devices: - /dev/galcore:/dev/galcore` | The only way into the kernel driver ([27.3](#273-every-module-library-and-driver)). On our device it is mode 0666, so no extra permission is needed |
| The device's NPU libraries, read-only | `volumes: - /usr/lib:/host/usr/lib:ro` | The entrypoint links only the NPU libraries and the `tflite_runtime` package from it ([27.4](#274-the-entrypoint-borrow-only-the-npu-libraries)). `:ro` makes the device's files unchangeable from inside |
| A volume for compiled NPU graphs | `npu-graph-cache:/var/cache/npu` and `VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1`, `VIV_VX_CACHE_BINARY_GRAPH_DIR=/var/cache/npu` | Evidence (the `.nb` file) and a faster restart ([30.3](#303-the-evidence-ladder)) |
| The switch | `USE_NPU=true` (`false` for the negative control) | Read by `InferenceEngine` ([29.1](#291-inferenceengine-try-the-npu-fall-back-remember-why)) |
| Native log lines | `TF_CPP_MIN_LOG_LEVEL=0` | Keeps TFLite's INFO lines, which carry delegation evidence ([29.2](#292-catching-the-c-evidence)) |
| The broker on port 1884 as well as 1883 | `mqtt-broker` `ports: - "1884:1883"` | The FactoryTalk Optix project's `MQTTClient1` connects to `127.0.0.1:1884` |

`docker-compose.npu.yml` adds exactly these lines to Part 1's `docker-compose.yml`. Nothing else is needed: no extra capabilities, no host network, nothing installed. The device must already *have* its NPU stack in `/usr/lib` (the **Device** rows of [27.3](#273-every-module-library-and-driver)); on our OptixEdge it does.

**The image travels as a file.** `docker save` (or `podman save`) writes an image with all its layers into one `.tar` archive, and `docker load` (or Portainer's import) unpacks it into the device's image store. On our OptixEdge that store is on `/persistent`; the root file system had only 20 MB free, so never copy the archive to `/`.

There are two ways to deploy: Portainer in a browser, or the `docker` command on your PC talking to the device.

### 33.2 Option A: Portainer

Portainer is a web interface for a Docker daemon. A Portainer **stack** is a compose file pasted into a form. Portainer accepts only one file, so the stack below is Part 1's `docker-compose.yml` and `docker-compose.npu.yml` already merged.

1. **Images → Import**: upload the image archive. Images saved by Podman keep the `localhost/` prefix: `localhost/anomaly_detection-detector:npu`. Import `eclipse-mosquitto:2` the same way if the device cannot reach Docker Hub.
2. **Stacks → Add stack**: name it `anomaly_detection`, choose *Web editor*, paste the YAML below, then **Deploy the stack**.
3. **Containers → the detector → Logs**: check the lines of [33.4](#334-checking-the-deployment).

```yaml
services:
  mqtt-broker:
    image: eclipse-mosquitto:2
    ports:
      - "1883:1883"   # MQTT for the detector and other clients
      - "1884:1883"   # the same broker, for the FactoryTalk Optix project's MQTTClient1
      - "9001:9001"   # reserved for MQTT over WebSocket
    command:
      - /bin/sh
      - -c
      - |
        cat > /tmp/mosquitto.conf <<EOF
        listener 1883 0.0.0.0
        allow_anonymous true
        listener 9001 0.0.0.0
        protocol websockets
        EOF
        exec mosquitto -c /tmp/mosquitto.conf
    healthcheck:
      test: ["CMD-SHELL", "mosquitto_sub -h 127.0.0.1 -p 1883 -t '$$SYS/broker/version' -C 1 -W 3 >/dev/null 2>&1"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    restart: unless-stopped

  detector:
    image: localhost/anomaly_detection-detector:npu
    pull_policy: never
    platform: linux/arm64
    devices:
      - /dev/galcore:/dev/galcore
    volumes:
      - /usr/lib:/host/usr/lib:ro
      - npu-graph-cache:/var/cache/npu
    environment:
      - USE_NPU=true
      - TF_CPP_MIN_LOG_LEVEL=0
      - VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1
      - VIV_VX_CACHE_BINARY_GRAPH_DIR=/var/cache/npu
    restart: unless-stopped
    depends_on:
      mqtt-broker:
        condition: service_healthy

volumes:
  npu-graph-cache:
```

For the negative control, edit the stack, set `USE_NPU=false` and click *Update the stack*. Portainer recreates the detector with the new value.

### 33.3 Option B: the Docker CLI from your PC

The `docker` command is only a **client**. Every command (`load`, `compose up`, `logs`) is turned into a request to a **daemon**, the program that actually runs containers. The environment variable `DOCKER_HOST` says which daemon to talk to. Point it at the device, and every command on your PC acts on the device:

```bash
# Linux / macOS / WSL
export DOCKER_HOST=tcp://optix_edge_ip:2375
```

```powershell
# Windows PowerShell
$env:DOCKER_HOST = "tcp://optix_edge_ip:2375"
```

Then, from `anomaly_detection/`:

```bash
docker load -i anomaly_detection-npu.tar        # streams the archive from the PC into the device's image store
docker compose -p anomaly_detection -f docker-compose.yml -f docker-compose.npu.yml up -d --no-build
docker compose -p anomaly_detection -f docker-compose.yml -f docker-compose.npu.yml logs -f detector
```

**The detail that surprises everyone.** Compose reads the two YAML files on your PC, but every path inside them (`/dev/galcore`, `/usr/lib`, the volume) is handed to the daemon, and the daemon resolves it **on the device**. `/usr/lib` means the OptixEdge's `/usr/lib`, even when you typed the command on Windows. That is exactly what we want here, but it also means a path to a file on your PC would not work.

`--no-build` tells compose to use the imported image and never try to build one (the files also contain a `build:` section, used only on a build machine).

**Security, before you open the port.** Port 2375 is plain HTTP with no password. Anyone who can reach it can start any container with any device or folder of the host mounted, which on a normal (rootful) daemon means full control of the device. Use 2375 only on an isolated lab network, and close it afterwards. For anything else, use the encrypted port **2376** with client certificates (`DOCKER_HOST=tcp://optix_edge_ip:2376`, `DOCKER_TLS_VERIFY=1`, `DOCKER_CERT_PATH=<folder with ca.pem, cert.pem, key.pem>`). On the OptixEdge, the remote Docker API and its TLS settings are managed by the device's System Manager (Docker page).

**On our test device**, the daemon is *rootless* (it runs as the user `docker`, uid 985) and listens only on a local socket (`unix:///run/user/985/docker.sock`), not on 2375 or 2376. So for the measurements in this part we ran the same `docker load` and `docker compose` commands in an SSH session on the device, with `DOCKER_HOST` set to that socket. The commands and the result are the same; only the road to the daemon differs.

### 33.4 Checking the deployment

The detector's first log lines come from the entrypoint, before Python starts:

```text
[npu] linked 20 host libraries from /host/usr/lib into /opt/npu/lib
[npu] every dependency of /opt/npu/lib/libvx_delegate.so resolves
[npu] using the host's tflite_runtime (/host/usr/lib/python3.12/site-packages/tflite_runtime)
[npu] /dev/galcore is present
[INFO] anomaly-detector: Model autoencoder_int8.tflite (int8) on NPU via tflite_runtime 2.16.2, warm-up 18.8 ms
[INFO] mqtt_client: Published status to anomaly/paperclip/status: accelerator=npu
```

Four `[npu]` lines, four facts: the NPU libraries were found, nothing they need is missing, the runtime is the device's own, and the door to the kernel driver is open. A warm-up under 20 ms means the compiled graph was read from the cache volume instead of being compiled again.

Then the detector itself reports the engine. After that, the proof moves to the HMI: the badge, the delegation line, the `.nb` file and a benchmark ([32.3](#323-reading-the-panel)). Then the negative control: with `USE_NPU=false` the badge must turn CPU, the delegation line must disappear and the benchmark must say `NPU not requested`.

For a one-shot check without MQTT, the image contains `tools/npu_check.py`. It uses the production code (`InferenceEngine` + `run_benchmark`), loads the model, runs 200 inferences on each engine and prints a verdict:

```bash
docker run --rm --device /dev/galcore -v /usr/lib:/host/usr/lib:ro localhost/anomaly_detection-detector:npu python tools/npu_check.py
```

If it prints `RESULT: inference ran on the NPU`, the detector will find the NPU too.

### 33.5 Rollback

Every object can be removed, and the device is then exactly as before:

- Portainer: *Stacks → anomaly_detection → Delete*, then *Images* and *Volumes* to remove `localhost/anomaly_detection-detector:npu` and `anomaly_detection_npu-graph-cache`.
- CLI: `docker compose -p anomaly_detection -f docker-compose.yml -f docker-compose.npu.yml down`, then `docker rmi localhost/anomaly_detection-detector:npu` and `docker volume rm anomaly_detection_npu-graph-cache`.

### 33.6 Troubleshooting

| Symptom (`NpuError`, or an `[npu]` log line) | Layer | What it means, what to do |
|---|---|---|
| `/host/usr/lib is not mounted` | Container config | Add `-v /usr/lib:/host/usr/lib:ro` |
| `/dev/galcore is missing` | Container config | Add `--device /dev/galcore` |
| `linked 0 host libraries` | Device | The device has no NPU libraries in `/usr/lib`: its firmware was built without them, or keeps them elsewhere (set `HOST_LIB_DIR`) |
| `no delegate at /opt/npu/lib/libvx_delegate.so` | Device | The device has the driver but not the VX delegate (a firmware without the ML packages). The detector runs on the CPU; see [27.7](#277-host-or-container-where-each-moving-part-belongs) |
| `no host tflite_runtime for python3.12` | Device ? image | The device has no `tflite_runtime` for the image's Python version. The image's LiteRT is used instead (runtime B of [27.5](#275-which-tflite-runtime)) |
| `the delegate needs libraries that are not available` | NPU libs | The listed library is missing or has another name on this device: add its pattern to `NPU_HOST_LIBS`. `NPU_DEBUG=1` prints the full `ldd` |
| `OSError: ... GLIBC_2.xx not found` | C runtime | The device's glibc is newer than the image's: use a newer base image ([27.6](#276-the-c-runtime-rule)) |
| `cannot enable executable stack` | C runtime | A library in the image is marked as needing an executable stack; the image clears that mark on LiteRT ([27.6](#276-the-c-runtime-rule)) |
| `undefined symbol: tflite_plugin_create_delegate` | Delegate | The file is not a TFLite external delegate; check `VX_DELEGATE_PATH` |
| Delegate loads, crash or error at the first invoke | Runtime ? delegate | Switch runtime: `HOST_TFLITE_RUNTIME=0` uses the image's LiteRT instead of the borrowed `tflite_runtime` ([27.5](#275-which-tflite-runtime)) |
| `Cannot connect to the Docker daemon` | Remote access | Wrong `DOCKER_HOST`, or the device does not expose the Docker API ([33.3](#333-option-b-the-docker-cli-from-your-pc)) |
| `no space left on device` during the import | Device storage | The archive or the image store is on a small partition; keep both on `/persistent` |
| Delegation `0/5` | Model | The delegate rejected every op: check that the int8 model is used (`precision: int8`) |
| Badge stays `?` in FactoryTalk Optix | MQTT | Check the plain-text subscription to `anomaly/paperclip/status` and the broker port 1884 ([32.2](#322-the-variables-in-modelnpu)) |

## 34. The Ladder Log: Predictions vs Results

The core learning loop: predict, measure, compare, explain the gap. Results from the OptixEdge unless marked.

| Question | Prediction | Reason | Result |
|---|---|---|---|
| Does int8 change any detection? | No | Error ≤ half a mark (0.002) on [0,1] inputs | ✅ 0 of 300 windows changed (laptop) |
| Does the image load on ARM with the CPU? | Yes | Same code as the amd64 test image | ✅ under QEMU on the laptop, and on the device |
| Does the delegate load inside the container? | Yes, if `ldd` resolves | Libraries lent from the same release as the driver | ✅ with the device's `tflite_runtime` (runtime A) and with the image's LiteRT 1.2.0 (runtime B) |
| Is the device's TFLite 2.19, as first reported? | Yes | The examples path given at the start | ❌ **2.16.2**: this device has `/usr/bin/tensorflow-lite-2.16.2/`, and `ldd` on the delegate and the device's `tflite_runtime` agree. Ask the device, not memory. The image therefore borrows the device's 2.16.2 runtime ([27.5](#275-which-tflite-runtime)) |
| How many nodes go to the NPU? | 5 of 5 | FULLY_CONNECTED and LOGISTIC are common int8 ops | ✅ 5 of 5 (`DELEGATE` node in the graph; OVXLIB log: 4 FCL2, 3 RELU, 1 SIGMOID) |
| Does the float32 model run on the NPU? | No, falls back | The NPU is an integer engine | ❌ Yes, in float16: 0.248 ms, outputs differ in the 7th decimal ([31](#31-why-the-npu-may-be-slower-here)) |
| Warm-up on the NPU | 100–1 000 ms | Graph compilation at the first invoke | ✅ better: 63–72 ms; 16 ms with the graph cache |
| p50 on the CPU (A53), `invoke()` only | 0.05–0.2 ms | Laptop 0.029 ms, A53 several times slower | ❌ better: **0.0106 ms** (XNNPACK) |
| p50 on the NPU, `invoke()` only | 0.2–1 ms | Fixed trip cost dominates 3 328 multiply-adds | ✅ **0.157 ms**, just below the range |
| Largest CPU vs NPU output difference | ≤ 0.004 | One output mark is 0.0039 | ✅ **0.0**: identical integers |
| The NPU On-time counter moves during a benchmark | Yes | The driver counts powered-on time | ❌ 601 ms once, 0 ms twice: not usable ([30.3](#303-the-evidence-ladder)) |
| The driver lists our container's process as an NPU client | Yes | It opened `/dev/galcore` | ✅ `python`, cgroup `docker-9c2e…`, memory on GPU 1 (NPU); empty with `USE_NPU=false` |
| Can the image carry TFLite, the delegate and TIM-VX itself, borrowing only the driver? | Yes | Above OpenVX the interface is a public standard ([27.7](#277-host-or-container-where-each-moving-part-belongs)) | ⏳ Partly proven: TIM-VX compiles and links against the device's driver release. The full image is not validated on the device yet |
| Do the newer OpenVX headers (6.4.15, bundled with TIM-VX) work with the device's 6.4.11 driver? | Yes | OpenVX is a public standard | ❌ `LD_BIND_NOW=1` found `vxTensorSelectLayer` and `vxInvalidateHandleVSI` missing; 15 capability switches differ. Headers from the 6.4.11 SDK fixed the link ([27.7](#277-host-or-container-where-each-moving-part-belongs)) |

## 35. Decision Log

| Decision | Alternatives | Why |
|---|---|---|
| Borrow the NPU stack from the device, read-only: Vivante driver, TIM-VX, VX delegate, TFLite and `tflite_runtime` | Image that carries everything above OpenVX ([27.7](#277-host-or-container-where-each-moving-part-belongs)); image built from the NXP BSP (path N2 of [19.4](#194-how-the-npu-will-be-used)); install packages on the device | Rule 1: nothing installed on the host. The device's libraries always match its kernel driver, and the design is measured end to end. The line at OpenVX is the recommended next step, once validated |
| Link only the NPU libraries, not the whole `/usr/lib` | `LD_LIBRARY_PATH=/host/usr/lib` | Two glibc or libstdc++ copies in one process crash in confusing ways ([27.4](#274-the-entrypoint-borrow-only-the-npu-libraries)) |
| The device's `tflite_runtime` as default runtime (A), LiteRT in the image as fallback (B) | LiteRT only | A uses the same `libtensorflow-lite.so.2.16.2` as the delegate: the pairing NXP tests ([27.5](#275-which-tflite-runtime)). B keeps a CPU runtime when the device has none |
| Debian trixie base, Python 3.12 | bookworm (Part 1), Python 3.11 | glibc must be ≥ the device's 2.39 ([27.6](#276-the-c-runtime-rule)); the borrowed `tflite_runtime` is compiled for CPython 3.12 |
| Clear the executable-stack mark on LiteRT | An older base image | glibc 2.41 refuses the mark; the library does not need it, and the image is safer without it |
| Keep the self-contained prototype out of the image | Ship it | Not validated end to end on the device. Its lesson, compile for the device's driver release, is recorded in [27.7](#277-host-or-container-where-each-moving-part-belongs) |
| Work with the device's rootless Docker | Switch to a rootful daemon | Nothing changed on the host. It also keeps debugfs closed to containers, so the driver evidence stays a root lab check |
| Driver client list as hardware evidence | The `gc/idle` On-time counter | The counter gave 601 ms once and 0 ms twice for the same benchmark; the client list was right every time ([30.3](#303-the-evidence-ladder)) |
| `GalcoreVersion` read from the lent `libGAL.so` when the kernel's version file is unreadable | Show `unknown` | On our device the kernel driver's version is only in debugfs (root-only). `libGAL` is built from the same driver release and carries its version string |
| A separate int8 file next to the float one | Replace float with int8 | The float model stays the CPU reference; `precision: auto` picks |
| Per-tensor weights | Per-channel (TFLite default) | NPU-friendly, and measured: zero decision changes |
| Fallback to the CPU on any NPU error | Fail hard | The detector must keep protecting the machine; the status explains why |
| Benchmark in a worker thread, on request | Benchmark at start-up | Never blocks the 1 Hz sensor path; the operator sees it happen |

---

# Reference

## Glossary

### General and anomaly-detection terms

| Term | Definition |
|---|---|
| **Activation function** | A mathematical function applied after each layer in a neural network to introduce non-linearity (e.g., ReLU, Sigmoid). |
| **Anomaly** | A data point that deviates significantly from the expected pattern. |
| **Autoencoder** | A neural network trained to reconstruct its input; used for anomaly detection by measuring reconstruction error. |
| **Batch** | A subset of training data processed together in one step. |
| **Bottleneck** | The narrow middle layer of an autoencoder that forces compression. |
| **Broker** | The central server in MQTT that routes messages between publishers and subscribers. |
| **Container** | A lightweight, portable package that includes an application and all its dependencies (powered by Docker). |
| **Edge computing** | Running computation on a device near the data source, rather than in a remote cloud. |
| **Epoch** | One full pass through the entire training dataset. |
| **Feature** | A single measurable property of the data (e.g., one sensor tag). |
| **Inference** | Using a trained model to make predictions on new data. |
| **Loss function** | A measure of how wrong the model's output is (we use MSE). |
| **MinMaxScaler** | A preprocessing step that rescales values to a fixed range (typically [0, 1]). |
| **MQTT** | Message Queuing Telemetry Transport — a lightweight pub/sub messaging protocol for IoT. |
| **MSE** | Mean Squared Error — average of squared differences between predicted and actual values. |
| **Normalisation** | Transforming data to a common scale so features are comparable. |
| **Optimizer** | An algorithm that adjusts model weights to reduce the loss (we use Adam). |
| **QoS** | Quality of Service — MQTT delivery guarantee level (0, 1, or 2). |
| **ReLU** | Rectified Linear Unit: max(0, x). A simple, effective activation function. |
| **Reconstruction error** | The difference between the autoencoder's input and output (our anomaly score). |
| **Sigmoid** | A function that squashes any value to the range [0, 1]. |
| **Three-sigma rule** | In a normal distribution, ~99.7% of data falls within 3 standard deviations of the mean. |
| **Topic** | An MQTT channel name that publishers send to and subscribers listen on. |
| **WebSocket** | A full-duplex communication protocol over a single TCP connection, supported natively by all modern browsers. |
| **Window** | A fixed-size group of consecutive data samples processed together. |
| **CSP (Content Security Policy)** | A browser security mechanism declared in an HTTP header or `<meta>` tag. It restricts which origins scripts may load from (`script-src`), which URLs the page may connect to (`connect-src`), etc. FTOptix's WebBrowser sets `default-src 'self'`, which means only same-origin resources are allowed. |
| **NetLogic** | An FTOptix concept: a C# class that inherits from `BaseNetLogic` and runs inside the FTOptix runtime. It has access to the project's OPC UA information model and can read/write variables, subscribe to events, and spin up tasks. |
| **PeriodicTask** | An FTOptix API class that executes a callback on a fixed time interval inside a NetLogic. Equivalent to a background timer thread but managed by the FTOptix runtime lifecycle. |
| **Digital Twin** | A virtual, real-time model of a physical object or process. In this project, the Three.js 3-D scene is a digital twin of the paper-clip machine: it reflects the machine's anomaly state visually. |

### Gesture-control terms

The terms of Part 2, grouped by topic and sorted alphabetically within each group. The link after each definition leads to the section that explains the term in full. Terms that the table above already defines, such as Broker, Inference, QoS and Topic, are not repeated.

**Pose and gesture rules**

| Term | Definition |
|---|---|
| **Confidence** | The number from 0 to 1 in a gesture event. It says how sure MoveNet was about the keypoints the rule used, not how likely it is that the operator meant the gesture ([13.8](#138-confidence-and-which-gesture-wins)). |
| **Cooldown** | The 1.0 s after any event during which no gesture can fire, with either hand (`cooldown_s`) ([14.1](#141-the-latch-and-its-release)). |
| **Detector** | The rule for one gesture. For each wrist with a new sample, the engine asks the detectors in a fixed order and takes the first answer ([13.8](#138-confidence-and-which-gesture-wins)). |
| **Heat map** | A grid of scores in which a high value means "probably here". MoveNet builds one for the centres of people and one for each of the 17 keypoint types ([10.5](#105-movenet)). |
| **Keypoint** | One of the 17 body points MoveNet finds, always in the same order; among them are the shoulders, elbows and wrists. Each comes as y, x and a score. The rules ignore keypoints with a score below 0.3 ([10.4](#104-pose-estimation-and-keypoints)). |
| **Latch** | The engine's memory of the last gesture. While it is set, no gesture fires. It works like a PLC one-shot (ONS) with a minimum pause added ([14.1](#141-the-latch-and-its-release)). |
| **Lightning, Thunder** | The two sizes of single-person MoveNet. Lightning reads 192 × 192 pictures and is "intended for latency-critical applications"; Thunder reads 256 × 256 pictures, for higher accuracy. We use Lightning ([10.5](#105-movenet)). |
| **Mirror** | The setting `mirror: true` flips x, so that positive x is the operator's own right ([13.3](#133-body-coordinates-in-shoulder-widths)). |
| **MoveNet** | Google's pose-estimation model, published in May 2021. We use SinglePose Lightning, int8, version 4 ([10.5](#105-movenet)). |
| **Pose estimation** | Finding where a person's body parts are in a picture ([10.4](#104-pose-estimation-and-keypoints)). |
| **Presence gate** | The three checks a frame must pass before it counts: both shoulders visible (score at least 0.3), at least 0.12 of the picture apart, and the point midway between them at most 0.35 from the middle (x between 0.15 and 0.85). A frame that fails any of them adds nothing to the tracks ([13.2](#132-the-presence-gate)). |
| **Release line** | `release_y`, −1.0: a line 1 SW below the shoulders. After the cooldown, the latch of a palm gesture is released only when the hand that made it is below this line or has not been seen for 0.5 s (`lost_s`). A swipe's latch is released right after the cooldown ([14.1](#141-the-latch-and-its-release)). |
| **Return stroke** | The movement back after a swipe. An opposite swipe that starts within 1.5 s of a swipe is treated as one and blocked (`opposite_swipe_block_s`) ([14.3](#143-the-return-stroke)). |
| **Sample** | One entry of a track: `t` (the time), `x` and `y` in shoulder widths, `quality`, `forearm_up` and `frame` (the number of the `update()` call) ([13.4](#134-a-track-for-each-wrist)). |
| **Score** | A number from 0 to 1 that says how sure the model is about a point ([10.4](#104-pose-estimation-and-keypoints)). |
| **Shoulder width (SW)** | The unit of the body coordinates: the distance between the operator's own shoulders. Zero is the middle of the shoulders, and y points up ([13.3](#133-body-coordinates-in-shoulder-widths)). |
| **SinglePose, MultiPose** | SinglePose finds one person: it picks the highest-scoring person centre, giving more weight to people near the middle of the frame. MultiPose "detects up to 6 poses". We use SinglePose ([10.5](#105-movenet)). |
| **Track** | The last 1.5 s of one wrist's samples. A new track starts when the previous frame did not see the wrist, when 0.3 s or more passed since the last sample (`max_gap_s`), or when the wrist moved faster than 15 SW per second (`max_wrist_speed`) ([13.4](#134-a-track-for-each-wrist)). |
| **Trimming** | Throwing away every sample of a track except the newest one (`_trim_to_newest`). It happens when a gesture fires, when the latch is released and when a return stroke is blocked, so every new gesture must be made in full ([14.2](#142-trimming)). |

**Model and runtime**

| Term | Definition |
|---|---|
| **COCO** | A public collection of photos in which people marked body points by hand. MoveNet was trained on it and on an internal Google dataset called Active ([10.2](#102-training-vs-inference-and-why-we-never-train)). |
| **Convolution** | A small window of weights that slides over the picture, multiplying and adding at each position ([10.1](#101-what-is-a-neural-network)). |
| **CPU** | The general-purpose processor. The i.MX 8M Plus has 4 Arm Cortex-A53 cores at up to 1.8 GHz, plus a separate Cortex-M7 core ([10.7](#107-cpu-npu-runtime-and-delegates)). |
| **Delegate** | A plug-in that lets the runtime hand parts of a model to an accelerator. Operations it cannot run stay on the CPU ([10.7](#107-cpu-npu-runtime-and-delegates)). |
| **Graph cache** | A folder that keeps the result of the NPU's graph compilation. NXP's comment on it: "to get the warmup time only once". Planned on a named volume, so that it survives container restarts ([19.4](#194-how-the-npu-will-be-used)). |
| **int8, uint8, float32** | Number types. int8 holds whole numbers from −128 to 127 and is used inside the model; uint8 holds 0 to 255 and is used for the input pixels; float32 holds 32-bit decimal numbers and is used for the output ([10.5](#105-movenet), [10.6](#106-quantization)). |
| **Interpreter** | The class that loads a model file and runs it. LiteRT, tflite-runtime and full TensorFlow all offer the same `Interpreter` class ([12.4](#124-running-movenet-and-reading-the-result)). |
| **Layer** | One stage of a neural network. Not to be confused with an image layer of a container ([10.1](#101-what-is-a-neural-network)). |
| **LiteRT** | Google's new name for TensorFlow Lite. Its Python package is `ai-edge-litert` ([10.7](#107-cpu-npu-runtime-and-delegates), [19.2](#192-why-litert)). |
| **Neural network** | A long chain of simple arithmetic, organized in layers ([10.1](#101-what-is-a-neural-network)). |
| **NPU** | Neural Processing Unit: a block of the chip built to do huge numbers of multiply-and-add operations in parallel. The i.MX 8M Plus NPU is rated at 2.3 TOPS ([10.7](#107-cpu-npu-runtime-and-delegates)). |
| **Per-tensor, per-channel** | Two ways to quantize: one scale for a whole tensor, or one scale for each output channel. In our model file, 146 tensors are per-channel and 143 use a single scale ([10.6](#106-quantization)). |
| **Probe** | One test `invoke()` right after the NPU delegate loads, so that a failure falls back to the CPU at startup. Not the same as `tools/npu_probe.sh`, which only reads information about the device ([12.4](#124-running-movenet-and-reading-the-result), [19.4](#194-how-the-npu-will-be-used)). |
| **Quantization** | Storing a model's numbers as 8-bit integers instead of 32-bit floats: a quarter of the space, and cheaper arithmetic ([10.6](#106-quantization)). |
| **Runtime** | The program that loads a model file and executes it ([10.7](#107-cpu-npu-runtime-and-delegates)). |
| **Scale, zero point** | The two settings of a quantized tensor: real value = (stored integer − zero point) × scale. Think of a ruler with 256 marks: the scale is the distance between two marks, and the zero point says which mark means 0 ([10.6](#106-quantization)). |
| **Tensor** | A block of numbers, such as the model's input picture or its output ([10.6](#106-quantization)). |
| **TOPS** | Trillion operations per second: a best-case peak figure ([10.7](#107-cpu-npu-runtime-and-delegates)). |
| **VX delegate** | NXP's delegate for the i.MX 8M Plus NPU, a library at `/usr/lib/libvx_delegate.so` on NXP's Linux images ([10.7](#107-cpu-npu-runtime-and-delegates)). |
| **Warm-up** | Three inferences on a black picture before the camera starts. On the NPU, the first one compiles the network for the hardware ([12.4](#124-running-movenet-and-reading-the-result)). |
| **Weights** | The numbers a network multiplies by, also called parameters. Training sets them; our service only uses them ([10.1](#101-what-is-a-neural-network), [10.2](#102-training-vs-inference-and-why-we-never-train)). |
| **XNNPACK** | A Google library of fast CPU routines. LiteRT creates it by itself and logs "Created TensorFlow Lite XNNPACK delegate for CPU." ([23.3](#233-run-the-unit-tests)). |

**Video and camera**

| Term | Definition |
|---|---|
| **BGR, RGB** | Two orders for the three colours of a pixel. OpenCV stores blue, green, red (BGR); MoveNet expects red, green, blue (RGB), so the service swaps them ([10.3](#103-what-a-computer-sees-in-an-image)). |
| **Decoding** | Turning the compressed data from the camera back into pixels. It costs CPU for every frame the camera sends, including the frames the service then skips ([10.8](#108-video-basics)). |
| **DirectShow** | One of the Windows camera interfaces. With it, the laptop webcam opens in about a second; the default Media Foundation backend can take much longer ([18.3](#183-why-the-webcam-needs-a-bridge)). |
| **FFmpeg** | A widely used video library that OpenCV can use as its backend. The service opens addresses and files through it, with a 5 s open timeout and a 2 s read timeout ([17.2](#172-camera-loss-and-reconnect)). |
| **Frame, fps** | A frame is one picture of a video; fps (frames per second) says how many arrive each second ([10.8](#108-video-basics)). |
| **H.264** | A video format that stores mostly the differences between frames: much less bandwidth than MJPEG, but a more complex decoder ([10.8](#108-video-basics)). |
| **MJPEG** | Motion JPEG: a video stream in which every frame is a separate JPEG picture. The webcam bridge serves one over HTTP ([10.8](#108-video-basics)). |
| **OpenCV** | The library that reads the camera. `cv2.VideoCapture` accepts a webcam number, an HTTP MJPEG address, an RTSP address or a file ([10.8](#108-video-basics)). |
| **Pixel** | One dot of a picture: three numbers (red, green, blue), each from 0 to 255 ([10.3](#103-what-a-computer-sees-in-an-image)). |
| **Rate cap** | The limit of 20 frames per second (`target_fps`). After each frame, the loop waits for the rest of the 0.05 s period, like a metronome ([12.2](#122-the-rate-cap)). |
| **RTSP** | Real Time Streaming Protocol: the protocol IP cameras use for video, with addresses that start with `rtsp://` ([10.8](#108-video-basics)). |
| **Substream** | A camera's low-resolution stream, offered next to the full-resolution main stream. Planned for the device, for example 640 × 360 at about 20 fps ([11.1](#111-production-on-the-imx-8m-plus), [19.5](#195-bottlenecks)). |
| **VPU** | The hardware video decoder of the i.MX 8M Plus, for H.265, H.264, VP9 and VP8 up to 1080p at 60 fps. The service does not use it ([10.8](#108-video-basics)). |
| **Webcam bridge** | `tools/webcam_bridge.py`: a small program that runs in normal Windows Python and serves the webcam as MJPEG over HTTP. It is needed because WSL2 does not see the laptop's USB devices ([18.3](#183-why-the-webcam-needs-a-bridge)). |

**MQTT messages**

| Term | Definition |
|---|---|
| **Client ID** | The name a client gives the broker when it connects. It must be unique: when a second client connects with the same ID, the broker disconnects the first. The service uses `gesture-recognizer` ([15.1](#151-topics)). |
| **Heartbeat** | The status message the service publishes every 2 s as a sign of life ([15.1](#151-topics), [15.4](#154-retained-status-and-the-last-will)). |
| **Idempotent** | Giving the same result whether a message arrives once or twice ([15.3](#153-qos-1-and-duplicates)). |
| **ISO 8601, UTC** | The timestamp format of the messages, for example `2026-10-03T06:25:06.883Z`. The final `Z` means UTC, Coordinated Universal Time ([15.2](#152-payloads)). |
| **Keepalive** | The 15 s interval the service gives the broker when it connects. After 1.5 times that, 22.5 s, without a word from the service, the broker publishes the last will. This case was not measured ([15.4](#154-retained-status-and-the-last-will)). |
| **Last will (LWT)** | A message the client hands the broker when it connects. The broker publishes it if the connection ends without a `DISCONNECT`. Ours is a retained `{"v": 1, "state": "offline"}` ([15.4](#154-retained-status-and-the-last-will)). |
| **Retained message** | The latest message on a topic, kept by the broker and handed to every new subscriber. The status is retained; the events are not ([15.4](#154-retained-status-and-the-last-will)). |
| **Service state** | The `state` field of the status: `starting`, `running`, `camera_lost` or `offline` ([17.1](#171-service-states)). |
| **TLS** | Transport Layer Security: an encrypted connection. A user name and password over TLS is one fix for the open broker ([25.2](#252-limitations)). |
| **UUID** | A random 36-character identifier. Every event gets a new one, in `event_id` ([15.2](#152-payloads)). |

**Health and timing**

| Term | Definition |
|---|---|
| **Exit code** | The number a program returns when it ends: 0 means success, any other value means failure. A healthcheck that exits with 0 reports a healthy container ([17.3](#173-heartbeat-file-and-container-healthcheck)). |
| **Exponential backoff** | Waiting longer after each failed retry: 1, 2, 4 and 8 s, then 10 s at most. Like calling back a busy phone number ([17.2](#172-camera-loss-and-reconnect)). |
| **Healthcheck** | A command the container runtime runs inside the container. Ours fails when the heartbeat file is more than 10 s old; three failures in a row mark the container unhealthy ([17.3](#173-heartbeat-file-and-container-healthcheck)). |
| **Heartbeat file** | `/tmp/gesture_heartbeat` inside the container. Nobody reads its content; its modification time tells when the main loop last passed the point that writes it ([17.3](#173-heartbeat-file-and-container-healthcheck)). |
| **Latency** | The time from the moment a frame arrives until the decision is made ([19](#19-performance-and-the-npu-plan)). |
| **ms, µs** | A millisecond (ms) is a thousandth of a second; a microsecond (µs) is a thousandth of a millisecond ([19.1](#191-measured-on-the-laptop)). |
| **p50, p95** | p50 is the median: half of the frames take less time. p95 is the time that 95 % of the frames beat ([11.4](#114-one-frames-journey)). |
| **Restart policy** | The rule that tells the container runtime when to start a container again. `restart: unless-stopped` acts when a container exits, not when it is unhealthy ([17.3](#173-heartbeat-file-and-container-healthcheck)). |
| **Thread** | An independent line of work that shares the program's memory. The service uses four ([11.3](#113-inside-the-service-four-threads)). |
| **Watchdog** | The thread that ends the process when the main loop has not fed it for 15 s (120 s at startup). The restart policy should then start the container again; that has not been tested ([17.4](#174-internal-watchdog)). |

**Containers and networking**

| Term | Definition |
|---|---|
| **Buildah** | The tool Podman uses to build images. It drops a Dockerfile's `HEALTHCHECK` when it builds an OCI image ([17.3](#173-heartbeat-file-and-container-healthcheck)). |
| **Compose file** | A YAML file that describes how to build and run containers: build, image, network mode, environment, healthcheck and restart policy ([18.2](#182-the-compose-file)). |
| **Compose provider** | The program that actually reads compose files. Podman's documentation calls `podman compose` "a thin wrapper around an external compose provider such as docker-compose or podman-compose" ([23.1](#231-prerequisites)). |
| **Docker** | A tool that packages an application with everything it needs into containers, explained in [2.4](#24-what-is-docker-briefly). Podman runs the same images and compose files ([18](#18-containers-and-the-laptop-setup)). |
| **Dockerfile** | The recipe for a container image: base image, `pip install`, `COPY`, `CMD` ([18.1](#181-the-image)). |
| **Environment variable** | A named setting handed to a program from outside. `CAMERA_URL`, `MQTT_HOST`, `MQTT_PORT` and `USE_NPU`, when set and not empty, replace `camera.url`, `mqtt.broker_host`, `mqtt.broker_port` and `model.use_npu` ([24](#24-tuning-the-gesture-parameters)). |
| **`host.containers.internal`** | A name that containers can use to reach their host. With Podman on Windows, it points at the Podman machine, not at Windows ([18.4](#184-networking-between-windows-podman-and-containers)). |
| **Image** | The packaged files a container runs from. Ours holds a small Debian Linux with Python, the libraries, the code and the model. It can be saved to a file and imported on the device ([18.1](#181-the-image)). |
| **Image layer** | One step of an image build. A layer is rebuilt only when its inputs change, which is why the library list is copied before the code ([18.1](#181-the-image)). |
| **localhost** | The name a machine uses for itself. Windows, the Podman machine and each container all have their own ([18.4](#184-networking-between-windows-podman-and-containers)). |
| **NAT** | Network address translation: rewriting addresses as traffic passes between networks. It can block RTSP video sent over UDP ([11.1](#111-production-on-the-imx-8m-plus)). |
| **Network mode** | How a container connects to the network. On the laptop: `bridge`, Podman's default network ([18.2](#182-the-compose-file)). Planned on the device: `host`, which shares the device's network, so the broker is at `127.0.0.1:1883` ([11.1](#111-production-on-the-imx-8m-plus)). |
| **OCI** | Open Container Initiative: the open standard for container images. Its image format has no healthcheck field ([17.3](#173-heartbeat-file-and-container-healthcheck)). |
| **Podman** | A container tool whose commands mirror Docker's. This project uses it on the laptop ([18](#18-containers-and-the-laptop-setup)). |
| **Podman machine** | The small Linux virtual machine that Podman creates with WSL2 on Windows. The containers run inside it ([18](#18-containers-and-the-laptop-setup)). |
| **Port, published port** | A port is a numbered door for network connections. When a container publishes a port, such as the broker's `"1883:1883"`, Podman forwards it to Windows ([18.4](#184-networking-between-windows-podman-and-containers)). |
| **Portainer** | A web interface for managing containers. Part 1 uses it on the edge device to import images (**Images** → **Import**), add stacks (**Stacks** → **Add stack**) and check logs ([2B](#2b-deploy-on-edge-device-with-portainer-stack)). |
| **WSL adapter address** | The address of Windows on the virtual adapter `vEthernet (WSL…)`. Only this PC can reach it, and it can change after a reboot ([18.4](#184-networking-between-windows-podman-and-containers)). |
| **WSL2** | Windows Subsystem for Linux, version 2. The Podman machine runs on it. It does not see the laptop's USB devices ([18](#18-containers-and-the-laptop-setup), [18.3](#183-why-the-webcam-needs-a-bridge)). |
| **x86-64, ARM64** | Two processor families with different instruction sets. The laptop is x86-64 (`linux/amd64`: 64-bit Intel and AMD); the device is ARM64. An image is built for one of them: here, the laptop could not run the ARM64 Mosquitto image ([18.1](#181-the-image), [23.4](#234-start-an-mqtt-broker)). |

**FactoryTalk Optix**

| Term | Definition |
|---|---|
| **Dialog** | A popup window, defined as a `DialogType`. `UICommands.OpenDialog` opens it in one session ([16](#16-factorytalk-optix-integration-planned)). |
| **Dynamic link** | A connection that makes a variable or property follow another one: change the source, and the linked value follows ([16](#16-factorytalk-optix-integration-planned)). |
| **GestureEventManager** | The planned NetLogic that reads the gesture and status messages. It checks each event, then opens the dialog, shows the message or changes the page, and it watches the service's heartbeat ([16.2](#162-the-gestureeventmanager-netlogic)). |
| **Model variable** | A named value in the project's `Model` folder. Screens can show it; logic can read and write it ([16](#16-factorytalk-optix-integration-planned)). |
| **NodeId** | The unique identifier of a node: a variable, an object, a screen ([16](#16-factorytalk-optix-integration-planned)). |
| **PanelLoader** | A container that shows one of several panels at a time. Its `Panel` property, a NodeId, says which one; its `ChangePanel` method switches to another ([16](#16-factorytalk-optix-integration-planned)). |
| **`PlainTextSubscription`** | Part of the MQTT client built into FactoryTalk Optix since version 1.6. It writes the text of each message on one topic into a Model variable ([16](#16-factorytalk-optix-integration-planned)). |
| **Session** | One running copy of the HMI user interface. The Web Presentation Engine can have several at once, for example one per connected browser ([16](#16-factorytalk-optix-integration-planned)). |
| **Web Presentation Engine** | The part of FactoryTalk Optix that shows the HMI in web browsers. In this project it also serves the 3-D twin page on port 8080 ([4](#4-architecture--data-flow), [16](#16-factorytalk-optix-integration-planned)). |

**The i.MX 8M Plus and its NPU software**

| Term | Definition |
|---|---|
| **BSP** | Board Support Package: the set of operating system, drivers and libraries that NXP provides for its processors. Libraries from a different release than the driver are treated as unsupported; a mismatch has not been tried ([18.1](#181-the-image), [19.4](#194-how-the-npu-will-be-used)). |
| **C runtime** | glibc and libstdc++, the basic libraries almost every Linux program is built on. A library compiled on one system generally needs a C runtime at least as new as the one it was built with ([19.4](#194-how-the-npu-will-be-used)). |
| **Device file** | A file under `/dev` through which programs talk to hardware. The probe assumes that the NPU is reached through `/dev/galcore` and checks for it first ([19.4](#194-how-the-npu-will-be-used)). |
| **galcore** | A kernel driver that NXP's build recipe describes as the "Kernel loadable module for Vivante GPU". The probe assumes that it also drives the NPU ([19.4](#194-how-the-npu-will-be-used)). |
| **i.MX 8M Plus** | The NXP chip of the production device: 4 Cortex-A53 cores, a Cortex-M7, an NPU and a VPU ([10.7](#107-cpu-npu-runtime-and-delegates), [10.8](#108-video-basics)). |
| **OpenVX, GAL** | Libraries below TIM-VX that turn the work into commands for the hardware. The probe checks for eight NPU libraries, these two among them ([19.4](#194-how-the-npu-will-be-used)). |
| **TIM-VX** | The VeriSilicon library the VX delegate is built on, "to facilitate deployment of Neural-Networks on Verisilicon ML accelerators" ([19.4](#194-how-the-npu-will-be-used)). |
| **Yocto** | The Yocto Project: tools for building a custom Linux system from recipes, such as the NXP recipes linked in 19.4. Building the image of path N2 needs NXP's Yocto build setup ([19.4](#194-how-the-npu-will-be-used)). |

**Python code and tests**

| Term | Definition |
|---|---|
| **Dataclass** | A Python class that mainly holds named values. `GestureConfig` is one; it is frozen, so its values cannot change after it is created ([20.2](#202-the-most-important-lines)). |
| **Fake** | A stand-in written for the tests, for example in place of a real camera ([21.1](#211-synthetic-poses-and-injected-time)). |
| **Injected time** | The time is passed into the engine instead of being read from the clock, so a test can play 10 s of poses in a few milliseconds ([13](#13-from-keypoints-to-gestures)). |
| **Mock** | A temporary replacement made with `unittest.mock.patch.object`, which swaps a name inside a module for one `with` block. The tests use it to make the NPU fail ([21.1](#211-synthetic-poses-and-injected-time)). |
| **Mutation check, mutant** | Breaking the code on purpose, one small change at a time, to see whether the tests notice. Each broken version is a mutant: in the run with 14 mutants, 12 were caught ([21.3](#213-test-first-and-the-mutation-check)). |
| **pip** | Python's package installer. `pip install -r` installs every package listed in a file ([18.1](#181-the-image), [23.1](#231-prerequisites)). |
| **Regular expression** | A pattern for searching text. `redact()` uses one to hide the user name and password in camera addresses ([20.2](#202-the-most-important-lines)). |
| **SHA-256** | A fingerprint (hash) of a file, 64 hexadecimal characters long. Checking it makes sure that every machine runs a byte-identical model file ([20.2](#202-the-most-important-lines)). |
| **Signal** | A short notice the operating system delivers to a program. Ctrl+C sends `SIGINT`; `podman stop` first sends `SIGTERM` ([20.2](#202-the-most-important-lines)). |
| **Stub** | A minimal stand-in. Stubs for the camera, the model and the publisher were used to check the watchdog ([17.4](#174-internal-watchdog)). |
| **Synthetic pose** | A pose made up by the test code instead of coming from a camera. All timings in sections 13 and 14 come from synthetic poses ([21.1](#211-synthetic-poses-and-injected-time)). |
| **Test-first** | Writing the tests before the code, also called test-driven development. The first run failed with an `ImportError`, as it should: the "red" step ([21.3](#213-test-first-and-the-mutation-check)). |
| **Traceback** | Python's error report: the chain of calls that led to the error ([23.3](#233-run-the-unit-tests)). |
| **Unit test** | A small automatic check of one piece of code. The service has 60, written with Python's `unittest` ([21](#21-testing-the-gesture-service)). |
| **Virtual environment** | A folder with its own set of installed Python packages, separate from the main Python installation ([23.1](#231-prerequisites)). |
| **Wheel** | A ready-built package file: pip copies it into place instead of compiling source code ([19.2](#192-why-litert)). |

## Sources

The outside pages that Part 2 relies on, grouped by topic. Each entry says what the page supports and links the sections that use it. Facts about the service itself come from its code, its tests and the measurements described in each section. Every link was checked and worked when this guide was written.

**MoveNet and pose estimation**

- [MoveNet blog post](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html) (TensorFlow blog, May 2021): the publication of MoveNet; Lightning and Thunder; training on COCO and Google's internal Active dataset; how the model works inside (MobileNetV2 with a feature pyramid network, heat maps, more weight for people near the frame centre); the cropping and smoothing of the browser version ([10.2](#102-training-vs-inference-and-why-we-never-train), [10.5](#105-movenet)).
- [TensorFlow MoveNet tutorial](https://www.tensorflow.org/hub/tutorials/movenet): the 17 keypoints, their order and their y, x, score format; Lightning "intended for latency-critical applications" and Thunder "intended for applications that require high accuracy"; padding a picture to a square ([10.4](#104-pose-estimation-and-keypoints), [10.5](#105-movenet), [12.3](#123-preparing-the-image-for-the-model), [22](#22-gesture-decision-log)).
- [TensorFlow.js MoveNet page](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/movenet/README.md): MultiPose "detects up to 6 poses"; the smoothing option of the browser version ([10.5](#105-movenet)).
- [MoveNet on Kaggle](https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8): the int8 SinglePose Lightning model and its Apache 2.0 license ([10.2](#102-training-vs-inference-and-why-we-never-train)).
- [NXP eIQ Model Zoo: MoveNet](https://github.com/NXP/eiq-model-zoo/blob/main/tasks/vision/pose-estimation/movenet/README.md): NXP's description of MoveNet and its MobileNetV2 network; the TensorFlow Hub download; "has been tested on i.MX 8MP", with no speed figure ([10.1](#101-what-is-a-neural-network), [10.5](#105-movenet)).
- [COCO dataset](https://cocodataset.org/): the public photo collection that MoveNet was partly trained on ([10.2](#102-training-vs-inference-and-why-we-never-train)).
- [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker): a palm detection model plus a hand landmarks model with 21 points, a candidate for a real open-palm check ([25.4](#254-adding-gestures-in-the-future)).

**LiteRT and quantization**

- [TensorFlow Lite is now LiteRT](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/) (Google announcement): the new name ([10.7](#107-cpu-npu-runtime-and-delegates), [19.2](#192-why-litert)).
- [LiteRT delegates](https://ai.google.dev/edge/litert/performance/delegates): what a delegate is; operations it cannot run stay on the CPU ([10.7](#107-cpu-npu-runtime-and-delegates)).
- [LiteRT 8-bit quantization specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec): scale and zero point; one scale per tensor or per channel ([10.6](#106-quantization)).
- [ai-edge-litert on PyPI](https://pypi.org/project/ai-edge-litert/) and [tflite-runtime on PyPI](https://pypi.org/project/tflite-runtime/): the newest releases, 2.2.0 and 2.14.0, and the ready-made wheels ([19.2](#192-why-litert)).
- [TensorFlow Lite benchmark tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/README.md): the `benchmark_model` options for timing each operation ([19.4](#194-how-the-npu-will-be-used)).
- [XNNPACK](https://github.com/google/XNNPACK): Google's library of fast CPU routines, used by TensorFlow Lite ([23.3](#233-run-the-unit-tests)).

**The i.MX 8M Plus and its NPU**

- [VAR-SOM-MX8M-PLUS board documentation](https://github.com/zephyrproject-rtos/zephyr/blob/main/boards/variscite/imx8mp_var_som/doc/index.rst) (Zephyr project): 4 Cortex-A53 cores at up to 1.8 GHz and a Cortex-M7 ([10.7](#107-cpu-npu-runtime-and-delegates)).
- [DART-MX8M-PLUS hardware documentation](https://github.com/peridio/docs/blob/main/src/docs-hardware/variscite/dart-mx8m-plus/index.mdx) (Peridio): the 2.3 TOPS NPU and the formats of the VPU ([10.7](#107-cpu-npu-runtime-and-delegates), [10.8](#108-video-basics)).
- [NXP EfficientDet demo](https://github.com/nxp-imx/efficientdet-imx/blob/lf-5.15.5_1.0.0/DEMO_README.md): the VX delegate library at `/usr/lib/libvx_delegate.so` ([10.7](#107-cpu-npu-runtime-and-delegates)).
- [NXP VX delegate README](https://github.com/nxp-imx/tflite-vx-delegate-imx/blob/lf-6.18.20_2.0.0/README.md): running `benchmark_model` with the VX delegate ([19.4](#194-how-the-npu-will-be-used)).
- [NXP VX delegate Python example `label_image.py`](https://github.com/nxp-imx/tflite-vx-delegate-imx/blob/lf-6.18.20_2.0.0/examples/python/label_image.py): `tflite_runtime` and `load_delegate` ([19.2](#192-why-litert)).
- [NXP LiteRT recipes](https://github.com/nxp-imx/meta-imx/tree/wrynose-6.18.20-2.0.0/meta-imx-ml/recipes-libraries/litert): release `lf-6.18.20_2.0.0` builds LiteRT 2.1.0 and a LiteRT VX delegate ([19.2](#192-why-litert)).
- [NXP galcore recipe](https://github.com/nxp-imx/meta-imx/blob/wrynose-6.18.20-2.0.0/meta-imx-bsp/recipes-kernel/kernel-modules/kernel-module-imx-gpu-viv_6.4.11.p4.6.bb): "Kernel loadable module for Vivante GPU", version 6.4.11.p4.6 ([19.4](#194-how-the-npu-will-be-used)).
- [NXP imx-gpu-viv recipes](https://github.com/nxp-imx/meta-imx/tree/wrynose-6.18.20-2.0.0/meta-imx-bsp/recipes-graphics/imx-gpu-viv): the user-space package that provides `libOpenVX`, with the same version as galcore ([19.4](#194-how-the-npu-will-be-used)).
- [TIM-VX](https://github.com/VeriSilicon/TIM-VX) (VeriSilicon): the library below the VX delegate ([19.4](#194-how-the-npu-will-be-used)).
- [NXP NNStreamer examples, `common_utils.sh`](https://github.com/nxp-imx/nxp-nnstreamer-examples/blob/main/common/common_utils.sh): the graph-cache settings and NXP's comments on them ([19.4](#194-how-the-npu-will-be-used)).
- [Avocado `imx8mp-npu-pose` reference](https://github.com/avocado-linux/references/blob/main/imx8mp-npu-pose/README.md): the NPU accelerates per-tensor int8 operations, while per-channel quantization falls back to the CPU; per-tensor quantization, "the NPU-optimal mode", collapses MoveNet's keypoint confidence to about 0; the first inference compiles the NPU graph, which takes seconds ([10.6](#106-quantization), [12.4](#124-running-movenet-and-reading-the-result), [19.4](#194-how-the-npu-will-be-used)).

**Video and cameras**

- [OpenCV image file reading and writing](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html): OpenCV's BGR colour order ([10.3](#103-what-a-computer-sees-in-an-image)).
- [OpenCV VideoCapture flags](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html): the open and read timeouts, which apply only to the FFmpeg and GStreamer backends ([17.2](#172-camera-loss-and-reconnect)).
- [OpenCV `VideoCapture` class](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html): a successful `set()` "doesn't ensure that the property value has been accepted by the capture device" ([18.3](#183-why-the-webcam-needs-a-bridge)).
- [OpenCV FFmpeg capture source](https://github.com/opencv/opencv/blob/4.x/modules/videoio/src/cap_ffmpeg_impl.hpp): the `key;value` format of `OPENCV_FFMPEG_CAPTURE_OPTIONS` ([19.5](#195-bottlenecks)).
- [FFmpeg formats documentation](https://ffmpeg.org/ffmpeg-formats.html): the `nobuffer` flag ([19.5](#195-bottlenecks)).
- [FFmpeg codecs documentation](https://ffmpeg.org/ffmpeg-codecs.html): the `low_delay` flag ([19.5](#195-bottlenecks)).
- [Connect USB devices to WSL](https://learn.microsoft.com/en-us/windows/wsl/connect-usb) (Microsoft): attaching USB devices to WSL2 with usbipd-win, which this project does not use ([18.3](#183-why-the-webcam-needs-a-bridge)).

**MQTT**

- [MQTT 3.1.1 specification](https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html) (OASIS): client IDs and the disconnection of a duplicate, QoS 1, retained messages, the will message, and the keepalive with its 1.5 × rule ([15.1](#151-topics), [15.3](#153-qos-1-and-duplicates), [15.4](#154-retained-status-and-the-last-will)).
- [paho-mqtt migration notes](https://github.com/eclipse-paho/paho.mqtt.python/blob/master/docs/migrations.rst): the new form of the callbacks in version 2.0, with `reason_code` and `properties` ([18.1](#181-the-image)).

**Containers**

- [Docker: start containers automatically](https://docs.docker.com/engine/containers/start-containers-automatically/): restart policies, described for containers that stop or exit ([17.3](#173-heartbeat-file-and-container-healthcheck)).
- [Docker: running containers, user](https://docs.docker.com/engine/containers/run/#user): root is the default user ([22](#22-gesture-decision-log)).
- [Docker: `compose up`](https://docs.docker.com/reference/cli/docker/compose/up/): Ctrl+C stops the containers with exit code 0 ([23.8](#238-stop-everything)).
- [Buildah `config.go`](https://github.com/containers/buildah/blob/main/config.go): `HEALTHCHECK` is dropped for OCI images ([17.3](#173-heartbeat-file-and-container-healthcheck)).
- [podman compose](https://docs.podman.io/en/latest/markdown/podman-compose.1.html): a wrapper around a compose provider; docker-compose takes precedence ([23.1](#231-prerequisites)).
- [podman build](https://docs.podman.io/en/latest/markdown/podman-build.1.html): the `--from` option ([23.9](#239-troubleshooting)).
- [podman machine set](https://docs.podman.io/en/latest/markdown/podman-machine-set.1.html): the `--user-mode-networking` option ([22](#22-gesture-decision-log)).
- [podman network create](https://docs.podman.io/en/latest/markdown/podman-network-create.1.html): the `isolate` option ([18.4](#184-networking-between-windows-podman-and-containers)).
- [podman pull](https://docs.podman.io/en/latest/markdown/podman-pull.1.html): the `--platform` option ([23.4](#234-start-an-mqtt-broker)).
- [podman stop](https://docs.podman.io/en/latest/markdown/podman-stop.1.html): `SIGTERM`, then `SIGKILL` after 10 s ([20.2](#202-the-most-important-lines)).

**Python**

- [What's new in Python 3.13](https://docs.python.org/3/whatsnew/3.13.html): `time.monotonic()` on Windows now uses the high-resolution clock ([12.1](#121-keeping-only-the-newest-frame)).
- [Python `sys.exit`](https://docs.python.org/3/library/sys.html#sys.exit): exit codes; a message is printed and gives exit code 1; the process ends only when it is called from the main thread ([17.3](#173-heartbeat-file-and-container-healthcheck), [17.4](#174-internal-watchdog), [20.2](#202-the-most-important-lines)).
- [Python `os._exit`](https://docs.python.org/3/library/os.html#os._exit): ends the process at once, without cleanup ([17.4](#174-internal-watchdog)).
- [Python `sys.platform`](https://docs.python.org/3/library/sys.html#sys.platform): `"win32"` on Windows ([20.2](#202-the-most-important-lines)).
- [Python `venv`](https://docs.python.org/3/library/venv.html): virtual environments, and the PowerShell execution policy for activating them ([23.1](#231-prerequisites), [23.9](#239-troubleshooting)).
- [PyYAML documentation](https://pyyaml.org/wiki/PyYAMLDocumentation): `safe_load` ([20.2](#202-the-most-important-lines)).

**FactoryTalk Optix**

- [NetLogic cheat sheet](https://github.com/FactoryTalk-Optix/NetLogic_CheatSheet): the MQTT client since version 1.6 and `PlainTextSubscription`; `VariableChange` handlers and the mandatory `-=` in `Stop()`; FactoryTalk Optix tasks instead of `async`/`await`; the warning for a `Start()` over 2000 ms; opening a dialog in every web session ([16](#16-factorytalk-optix-integration-planned), [16.2](#162-the-gestureeventmanager-netlogic)).
- [PanelLoader reference](https://help.optix.cloud.rockwellautomation.com/1.4.1.7/en/Reference/FTOptix-UI/ObjectTypes/PanelLoader.html) (FactoryTalk Optix help): the `Panel` property and the `ChangePanel` method ([16](#16-factorytalk-optix-integration-planned)).
- [Optix_Sample_PanelLoaderChangePanelNetLogic](https://github.com/FactoryTalk-Optix/Optix_Sample_PanelLoaderChangePanelNetLogic) (Rockwell sample): switching pages with `ChangePanel` from NetLogic, the fallback ([16.2](#162-the-gestureeventmanager-netlogic), [22](#22-gesture-decision-log)).
