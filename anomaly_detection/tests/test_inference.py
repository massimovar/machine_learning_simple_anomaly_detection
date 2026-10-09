import json
import os
import sys
import tempfile
import unittest

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
sys.path.insert(0, SRC)

import npu_evidence  # noqa: E402
from benchmark import MAX_RUNS, run_benchmark  # noqa: E402
from detector import DetectionResult, compute_anomaly_score, dequantize, detect, quantize  # noqa: E402
from model import InferenceEngine, delegation_summary, graph_delegation, resolve_model_settings  # noqa: E402

INT8_DETAIL = {"dtype": np.int8, "quantization": (1 / 255, -128)}
FLOAT_MODEL = os.path.join(MODELS, "autoencoder.tflite")
INT8_MODEL = os.path.join(MODELS, "autoencoder_int8.tflite")
MODEL_CFG = {
    "variants": {
        "float32": {"model_path": "f.tflite", "threshold_path": "f.json"},
        "int8": {"model_path": "q.tflite", "threshold_path": "q.json"},
    },
    "precision": "auto",
    "use_npu": False,
    "vx_delegate_path": "/usr/lib/libvx_delegate.so",
}


class QuantizationTests(unittest.TestCase):
    def test_round_trip_error_is_at_most_half_a_step(self):
        values = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        restored = dequantize(quantize(values, INT8_DETAIL), INT8_DETAIL)
        self.assertLessEqual(np.max(np.abs(restored - values)), 0.5 / 255 + 1e-6)

    def test_values_outside_the_range_saturate(self):
        q = quantize(np.array([-0.5, 1.5], dtype=np.float32), INT8_DETAIL)
        self.assertEqual(q.tolist(), [-128, 127])

    def test_float_tensors_pass_through(self):
        detail = {"dtype": np.float32, "quantization": (0.0, 0)}
        values = np.array([0.25, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(dequantize(quantize(values, detail), detail), values)


class ModelSettingsTests(unittest.TestCase):
    def test_auto_picks_float32_on_cpu_and_int8_on_npu(self):
        self.assertEqual(resolve_model_settings(MODEL_CFG, {})["precision"], "float32")
        npu = resolve_model_settings(MODEL_CFG, {"USE_NPU": "true"})
        self.assertTrue(npu["use_npu"])
        self.assertEqual((npu["precision"], npu["model_path"], npu["threshold_path"]), ("int8", "q.tflite", "q.json"))

    def test_env_overrides_config(self):
        cfg = dict(MODEL_CFG, use_npu=True)
        settings = resolve_model_settings(cfg, {"USE_NPU": "0", "MODEL_PRECISION": "int8", "VX_DELEGATE_PATH": "/x.so"})
        self.assertFalse(settings["use_npu"])
        self.assertEqual(settings["precision"], "int8")
        self.assertEqual(settings["vx_delegate_path"], "/x.so")

    def test_unknown_precision_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_model_settings(MODEL_CFG, {"MODEL_PRECISION": "int4"})


class EngineTests(unittest.TestCase):
    def test_float_and_int8_models_reconstruct_alike(self):
        vector = np.full(10, 0.5, dtype=np.float32)
        f_engine, q_engine = InferenceEngine(FLOAT_MODEL), InferenceEngine(INT8_MODEL)
        self.assertEqual((f_engine.precision, q_engine.precision), ("float32", "int8"))
        diff = np.max(np.abs(f_engine.reconstruct(vector) - q_engine.reconstruct(vector)))
        self.assertLess(diff, 0.05)
        self.assertIsNotNone(q_engine.inference_ms)

    def test_missing_delegate_falls_back_to_cpu(self):
        engine = InferenceEngine(INT8_MODEL, use_npu=True, vx_delegate_path="/nonexistent/libvx_delegate.so")
        info = engine.describe()
        self.assertEqual(info["accelerator"], "cpu")
        self.assertTrue(info["npu_requested"])
        self.assertIsNotNone(info["npu_error"])
        self.assertIsNone(info["delegate"])

    def test_detect_scores_against_the_float_input(self):
        engine = InferenceEngine(INT8_MODEL)
        vector = np.array([1.5] + [0.5] * 9, dtype=np.float32)  # 1.5 saturates in int8
        result = detect(engine, vector, threshold=0.01)
        expected = compute_anomaly_score(vector, engine.reconstruct(vector))
        self.assertAlmostEqual(result.score, expected, places=6)
        self.assertTrue(result.is_anomaly)


class EvidenceTests(unittest.TestCase):
    def test_delegation_lines_are_parsed(self):
        log = ["INFO: Created TensorFlow Lite XNNPACK delegate for CPU.",
               "INFO: Replacing 5 out of 5 node(s) with delegate (Vx Delegate) node, yielding 1 partitions."]
        self.assertEqual(delegation_summary(log), [{"delegate": "Vx Delegate", "nodes": 5, "of": 5}])

    def test_graph_delegation_counts_nodes_inside_each_partition(self):
        # Our model's shape: t0 -> FC -> FC -> FC -> FC -> LOGISTIC -> t5 (weights omitted)
        chain = [{"op_name": "FULLY_CONNECTED", "inputs": [i], "outputs": [i + 1]} for i in range(4)]
        chain.append({"op_name": "LOGISTIC", "inputs": [4], "outputs": [5]})
        whole = chain + [{"op_name": "DELEGATE", "inputs": [0], "outputs": [5]}]
        self.assertEqual(graph_delegation(whole, "Vx Delegate"), [{"delegate": "Vx Delegate", "nodes": 5, "of": 5}])
        # Sigmoid left on the CPU: the partition ends at t4
        partial = chain + [{"op_name": "DELEGATE", "inputs": [0], "outputs": [4]}]
        self.assertEqual(graph_delegation(partial, "Vx Delegate")[0]["nodes"], 4)
        self.assertEqual(graph_delegation(chain, "Vx Delegate"), [])

    def test_real_cpu_engine_reports_graph_evidence(self):
        delegation = InferenceEngine(INT8_MODEL).delegation()
        for entry in delegation:  # runtimes without XNNPACK have no DELEGATE node: empty is fine
            self.assertEqual(entry["of"], 5)

    def test_counters_parse_name_number_lines(self):
        text = "Start:   10 ns\nOn:      2500 ns\ngarbage line\nIdle = 7\n"
        self.assertEqual(npu_evidence.parse_counters(text), {"Start": 10, "On": 2500, "Idle": 7})

    def test_counters_parse_galcore_idle_file(self):
        # Real /sys/kernel/debug/gc/idle from the OptixEdge: thousands separators
        text = "On:                    8,711,857,271 ns\nOff:              20,209,908,361,955 ns\nIdle:                              0 ns\n"
        self.assertEqual(npu_evidence.parse_counters(text),
                         {"On": 8711857271, "Off": 20209908361955, "Idle": 0})

    def test_graph_cache_lists_nb_files_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "abc.nb"), "wb") as f:
                f.write(b"x" * 42)
            open(os.path.join(tmp, "other.txt"), "w").close()
            env = {"VIV_VX_ENABLE_CACHE_GRAPH_BINARY": "1", "VIV_VX_CACHE_BINARY_GRAPH_DIR": tmp}
            info = npu_evidence.graph_cache(env)
            self.assertEqual((info["files"], info["bytes"], info["names"]), (1, 42, ["abc.nb"]))
            self.assertEqual(npu_evidence.graph_cache({})["files"], 0)

    def test_counters_absent_without_path(self):
        self.assertIsNone(npu_evidence.read_counters({}))

    def test_driver_version_from_sysfs_first_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            sysfs = os.path.join(tmp, "version")
            with open(sysfs, "w") as f:
                f.write("6.4.11.p4.1190909 built at host\nPlatform path: ...\n")
            self.assertEqual(npu_evidence.galcore_version(paths=(sysfs,), gal_library=None), "galcore 6.4.11.p4.1190909")

    def test_driver_version_falls_back_to_lent_libgal(self):
        # Real marker inside the OptixEdge's /usr/lib/libGAL.so
        with tempfile.TemporaryDirectory() as tmp:
            lib = os.path.join(tmp, "libGAL.so")
            with open(lib, "wb") as f:
                f.write(b"\x7fELF\x00junk$VERSION$6.4.11:745085:cb782f4af5$\x00more")
            missing = os.path.join(tmp, "absent")
            self.assertEqual(npu_evidence.galcore_version(paths=(missing,), gal_library=lib), "libGAL 6.4.11:745085")
            self.assertIsNone(npu_evidence.galcore_version(paths=(missing,), gal_library=missing))


class BenchmarkTests(unittest.TestCase):
    def test_cpu_only_run_reports_why_npu_is_missing(self):
        main, cpu = InferenceEngine(INT8_MODEL), InferenceEngine(INT8_MODEL)
        result = run_benchmark(main, cpu, np.full(10, 0.5, dtype=np.float32), runs=20)
        self.assertIsNone(result["npu"])
        self.assertIn("USE_NPU", result["npu_unavailable"])
        self.assertEqual(result["cpu"]["runs"], 20)
        self.assertEqual(result["max_output_difference"], 0.0)
        json.dumps(result)  # must be publishable as-is

    def test_runs_are_capped(self):
        engine = InferenceEngine(INT8_MODEL)
        result = run_benchmark(engine, engine, np.zeros(10, dtype=np.float32), runs=10 ** 9)
        self.assertEqual(result["runs"], MAX_RUNS)


class AlertTests(unittest.TestCase):
    def test_core_fields_unchanged_and_evidence_appended(self):
        payload = json.loads(DetectionResult(True, 0.5, 0.01).to_alert_json({"accelerator": "npu"}))
        self.assertEqual(list(payload)[:3], ["is_anomaly", "anomaly_score", "threshold"])
        self.assertEqual(payload["accelerator"], "npu")
        self.assertEqual(json.loads(DetectionResult(False, 0.0, 0.01).to_alert_json()),
                         {"is_anomaly": False, "anomaly_score": 0.0, "threshold": 0.01})


if __name__ == "__main__":
    unittest.main()
