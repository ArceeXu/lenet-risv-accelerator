"""Integer LeNet reference, calibration, evaluation and hardware data export.

Run from the project root: python python/quantize.py --test-samples 100
Use --test-samples 10000 for the complete MNIST test set.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from model import LeNet


ROOT = Path(__file__).resolve().parents[1]
LAYER_NAMES = ("conv1", "conv2", "fc1", "fc2", "fc3")
ACTIVATION_NAMES = ("relu1", "relu2", "relu3", "relu4", "fc3")
INPUT_SCALE = 1.0 / 127.0
INT32 = np.iinfo(np.int32)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def checked_int32(values):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite accumulator or bias")
    if np.any(values < INT32.min) or np.any(values > INT32.max):
        raise OverflowError("INT32 accumulator or bias overflow")
    return values.astype(np.int32)


def quantize_int8(values, scale):
    return np.clip(np.rint(values / scale), -127, 127).astype(np.int8)


def fixed_multiplier(real_multiplier):
    if not np.isfinite(real_multiplier) or real_multiplier < 0:
        raise ValueError("Scale multiplier must be finite and nonnegative")
    shift = 30
    while shift < 62 and real_multiplier * (1 << shift) < (1 << 30):
        shift += 1
    while shift > 0 and real_multiplier * (1 << shift) > INT32.max:
        shift -= 1
    multiplier = int(np.rint(real_multiplier * (1 << shift)))
    if not 0 <= multiplier <= INT32.max:
        raise ValueError("Scale cannot be represented by an INT32 multiplier")
    return multiplier, shift


def requantize(accumulator, multipliers, shifts, relu):
    accumulator = checked_int32(accumulator)
    broadcast_shape = (len(multipliers),) + (1,) * (accumulator.ndim - 1)
    multipliers = multipliers.astype(np.int64).reshape(broadcast_shape)
    shifts = shifts.astype(np.int64).reshape(broadcast_shape)
    product = accumulator.astype(np.int64) * multipliers
    half = np.where(shifts > 0, np.left_shift(1, np.maximum(shifts - 1, 0)), 0)
    magnitude = np.right_shift(np.abs(product) + half, shifts)
    rounded = np.where(product < 0, -magnitude, magnitude)
    return np.clip(rounded, 0 if relu else -127, 127).astype(np.int8)


def convolution(inputs, parameters):
    weights = parameters["weight"]
    kernel_height, kernel_width = weights.shape[2:]
    windows = np.lib.stride_tricks.sliding_window_view(
        inputs, (kernel_height, kernel_width), axis=(1, 2)
    )
    accumulator = np.einsum(
        "chwkl,ockl->ohw", windows.astype(np.int64),
        weights.astype(np.int64), optimize=True
    )
    accumulator += parameters["bias"].astype(np.int64)[:, None, None]
    return checked_int32(accumulator)


def linear(inputs, parameters):
    accumulator = parameters["weight"].astype(np.int64) @ inputs.astype(np.int64)
    accumulator += parameters["bias"].astype(np.int64)
    return checked_int32(accumulator)


def max_pool(inputs):
    channels, height, width = inputs.shape
    if height % 2 or width % 2:
        raise ValueError("This reference requires even pooling dimensions")
    return inputs.reshape(channels, height // 2, 2, width // 2, 2).max(axis=(2, 4))


def integer_forward(raw_pixels, parameters):
    pixels = raw_pixels.astype(np.int32)
    inputs = ((pixels * 127 + 127) // 255).astype(np.int8)[None, :, :]
    outputs = {"input": inputs}
    for index, name in enumerate(LAYER_NAMES):
        layer = parameters[name]
        accumulator = convolution(inputs, layer) if index < 2 else linear(inputs, layer)
        outputs[name + "_acc"] = accumulator
        inputs = requantize(accumulator, layer["multiplier"], layer["shift"], index < 4)
        outputs[ACTIVATION_NAMES[index]] = inputs
        if index < 2:
            inputs = max_pool(inputs)
            outputs[f"pool{index + 1}"] = inputs
        if index == 1:
            inputs = inputs.reshape(-1)
            outputs["flatten"] = inputs
    return outputs


@torch.no_grad()
def calibrate(model, training_data, count):
    generator = np.random.default_rng(42)
    indices = generator.choice(len(training_data), count, replace=False)
    maxima = {name: 0.0 for name in ACTIVATION_NAMES}
    for start in range(0, count, 128):
        images = torch.stack([training_data[int(index)][0] for index in indices[start:start + 128]])
        outputs = model.forward_with_intermediates(images)
        for name in ACTIVATION_NAMES:
            maxima[name] = max(maxima[name], float(outputs[name].abs().max()))
    return {name: max(value / 127.0, 1e-12) for name, value in maxima.items()}


def build_parameters(model, activation_scales):
    parameters = {}
    input_scale = INPUT_SCALE
    for name, activation in zip(LAYER_NAMES, ACTIVATION_NAMES):
        module = getattr(model, name)
        weights = module.weight.detach().numpy().astype(np.float64)
        biases = module.bias.detach().numpy().astype(np.float64)
        axes = tuple(range(1, weights.ndim))
        weight_scales = np.maximum(np.max(np.abs(weights), axis=axes) / 127.0, 1e-12)
        broadcast_shape = (weights.shape[0],) + (1,) * (weights.ndim - 1)
        output_scale = activation_scales[activation]
        pairs = [fixed_multiplier(input_scale * scale / output_scale) for scale in weight_scales]
        parameters[name] = {
            "weight": quantize_int8(weights, weight_scales.reshape(broadcast_shape)),
            "bias": checked_int32(np.rint(biases / (input_scale * weight_scales))),
            "multiplier": np.array([pair[0] for pair in pairs], dtype=np.int32),
            "shift": np.array([pair[1] for pair in pairs], dtype=np.int32),
            "input_scale": input_scale,
            "output_scale": output_scale,
            "weight_scales": weight_scales.tolist(),
        }
        input_scale = output_scale
    return parameters


def export_array(directory, name, values, export_c=False):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / (name + ".npy"), values)
    bits = values.dtype.itemsize * 8
    mask = (1 << bits) - 1
    flat = values.reshape(-1)
    memory = "\n".join(f"{int(value) & mask:0{bits // 4}X}" for value in flat)
    (directory / (name + ".mem")).write_text(memory + "\n", encoding="ascii")
    if export_c:
        rows = [", ".join(str(int(value)) for value in flat[start:start + 16])
                for start in range(0, flat.size, 16)]
        contents = (
            "#pragma once\n#include <stdint.h>\n\n"
            f"static const int{bits}_t {name}[{flat.size}] = {{\n    "
            + ",\n    ".join(rows) + "\n};\n"
        )
        (directory / (name + ".h")).write_text(contents, encoding="ascii")


def export_parameters(parameters, output_directory, calibration_count):
    manifest = {
        "precision": "INT8 symmetric per-output-channel weights, per-tensor activations",
        "integer_range": [-127, 127],
        "zero_point": 0,
        "input_scale": INPUT_SCALE,
        "input_conversion": "(pixel * 127 + 127) // 255, pixel in [0, 255]",
        "parameter_rounding": "nearest, ties to even",
        "requantization_rounding": "nearest, ties away from zero",
        "requantization": "clip(round_away(acc * multiplier / 2**shift), lower, 127)",
        "relu_lower_bound": 0,
        "signed_lower_bound": -127,
        "accumulator_dtype": "INT32; overflow raises an error",
        "requantization_product_dtype": "INT64",
        "requantization_shift_range": [0, 62],
        "multiplier_underflow": "Values rounding to zero at shift 62 use multiplier zero",
        "layout": {"feature": "CHW", "conv_weight": "OIHW", "fc_weight": "OI", "flatten": "C"},
        "mem_format": "one element per line, hexadecimal two's complement, no header",
        "calibration_samples": calibration_count,
        "calibration_seed": 42,
        "layers": {},
    }
    for name, layer in parameters.items():
        metadata = {key: value for key, value in layer.items() if not isinstance(value, np.ndarray)}
        metadata["arrays"] = {}
        for field in ("weight", "bias", "multiplier", "shift"):
            values = layer[field]
            export_array(output_directory, name + "_" + field, values, export_c=True)
            metadata["arrays"][field] = {"shape": list(values.shape), "dtype": str(values.dtype)}
        manifest["layers"][name] = metadata
    write_json(output_directory / "quantization_manifest.json", manifest)


@torch.no_grad()
def evaluate(model, test_data, parameters, count, vector_count, result_directory):
    result_directory.mkdir(parents=True, exist_ok=True)
    fp32_correct = 0
    int8_correct = 0
    matches = 0
    with (result_directory / "prediction_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "label", "fp32_prediction", "int8_prediction"))
        for index in range(count):
            image, label = test_data[index]
            raw_pixels = test_data.data[index].numpy()
            fp32_outputs = model.forward_with_intermediates(image.unsqueeze(0))
            fp32_prediction = int(fp32_outputs["fc3"].argmax(dim=1).item())
            outputs = integer_forward(raw_pixels, parameters)
            int8_prediction = int(np.argmax(outputs["fc3"]))
            fp32_correct += fp32_prediction == label
            int8_correct += int8_prediction == label
            matches += fp32_prediction == int8_prediction
            writer.writerow((index, label, fp32_prediction, int8_prediction))
            if index < vector_count:
                directory = ROOT / "data" / "test_vectors" / "int8" / f"sample_{index:04d}"
                for name, values in outputs.items():
                    export_array(directory, name, values)
                np.save(directory / "input_uint8.npy", raw_pixels)
                for name, values in fp32_outputs.items():
                    np.save(directory / (name + "_fp32.npy"), values.squeeze(0).numpy())
                write_json(directory / "label.json", {
                    "label": label, "fp32_prediction": fp32_prediction,
                    "int8_prediction": int8_prediction,
                })
            if (index + 1) % 100 == 0 or index + 1 == count:
                print(f"Evaluated {index + 1}/{count}; INT8 accuracy {int8_correct / (index + 1):.2%}", flush=True)
    metrics = {
        "samples": count,
        "fp32_accuracy": fp32_correct / count,
        "int8_accuracy": int8_correct / count,
        "accuracy_drop_percentage_points": 100 * (fp32_correct - int8_correct) / count,
        "prediction_agreement": matches / count,
        "int8_correct": int8_correct,
        "note": "PC reference accuracy, not SoC validation or hardware performance",
    }
    write_json(result_directory / "quantization_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--export-vectors", type=int, default=10)
    arguments = parser.parse_args()
    if not 1 <= arguments.test_samples <= 10000:
        parser.error("--test-samples must be between 1 and 10000")
    if not 1 <= arguments.calibration_samples <= 60000:
        parser.error("--calibration-samples must be between 1 and 60000")
    if not 0 <= arguments.export_vectors <= arguments.test_samples:
        parser.error("--export-vectors must be between 0 and --test-samples")
    model = LeNet().cpu().eval()
    state = torch.load(ROOT / "data" / "checkpoints" / "lenet_fp32.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    training_data = datasets.MNIST(ROOT / "data" / "raw", train=True, download=False, transform=transforms.ToTensor())
    test_data = datasets.MNIST(ROOT / "data" / "raw", train=False, download=False, transform=transforms.ToTensor())
    print("Calibrating activation ranges on training data...", flush=True)
    scales = calibrate(model, training_data, arguments.calibration_samples)
    parameters = build_parameters(model, scales)
    export_parameters(parameters, ROOT / "data" / "weights" / "int8", arguments.calibration_samples)
    evaluate(model, test_data, parameters, arguments.test_samples, arguments.export_vectors,
             ROOT / "results" / "quantization")


if __name__ == "__main__":
    main()
