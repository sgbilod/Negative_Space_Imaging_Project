"""
Minimal Flask API for the negative space imaging pipeline.

POST /analyze  — multipart image upload → JSON analysis result
GET  /health   — liveness probe
"""

import base64
import io
import os
import sys
import tempfile

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(__file__))
from imaging_pipeline import run_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send a multipart field named 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    suffix = os.path.splitext(file.filename or "upload.png")[1] or ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as in_f:
        in_path = in_f.name
        file.save(in_path)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out_f:
        out_path = out_f.name

    try:
        result = run_pipeline(in_path, out_path)

        with open(out_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        return jsonify(
            {
                "regions": result["regions"],
                "negative_space_ratio": result["negative_space_ratio"],
                "enhanced_image_b64": img_b64,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
