"""Export all MNIST test images, labels and existing Python INT8 predictions."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]


def read_idx(name):
    directory = ROOT / "data" / "raw" / "MNIST" / "raw"
    path = directory / name
    if path.exists():
        return path.read_bytes()
    compressed = directory / (name + ".gz")
    with gzip.open(compressed, "rb") as stream:
        return stream.read()


def main():
    images = read_idx("t10k-images-idx3-ubyte")
    labels = read_idx("t10k-labels-idx1-ubyte")
    if len(images) != 16 + 10000 * 784 or struct.unpack(">IIII", images[:16]) != (2051, 10000, 28, 28):
        raise ValueError("Expected 10000 MNIST test images of size 28x28")
    if len(labels) != 8 + 10000 or struct.unpack(">II", labels[:8]) != (2049, 10000):
        raise ValueError("Expected 10000 MNIST test labels")
    report = ROOT / "results" / "quantization" / "prediction_comparison.csv"
    with report.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10000:
        raise ValueError("Run python/quantize.py --test-samples 10000 first")
    predictions = bytearray()
    for index, row in enumerate(rows):
        label = labels[8 + index]
        prediction = int(row["int8_prediction"])
        if int(row["index"]) != index or int(row["label"]) != label:
            raise ValueError(f"Python report index/label mismatch at {index}")
        if not 0 <= label <= 9 or not 0 <= prediction <= 9:
            raise ValueError(f"Invalid class at {index}")
        predictions.append(prediction)
    output = ROOT / "data" / "c_eval"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "images.idx": images,
        "labels.idx": labels,
        "python_int8_predictions.bin": bytes(predictions),
    }
    for name, contents in artifacts.items():
        (output / name).write_bytes(contents)
    manifest = {
        "samples": 10000,
        "python_prediction_source": str(report.relative_to(ROOT)),
        "python_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "files_sha256": {name: hashlib.sha256(contents).hexdigest() for name, contents in artifacts.items()},
        "note": "Predictions reuse the existing Python INT8 evaluation; no inference is rerun.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported 10000 images, labels and Python INT8 predictions to {output}")


if __name__ == "__main__":
    main()
