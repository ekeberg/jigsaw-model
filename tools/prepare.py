"""
Builds the on-device image generation model that the app downloads.

SD-Turbo (stabilityai/sd-turbo) is exported to ONNX, the text encoder and the UNet are quantized to 8 bits
(dynamic quantization, which ONNX Runtime's CPU provider runs on any arm64 phone), and the result is written to
OUT together with the tokenizer files and a manifest (model.json) the app reads: file sizes and SHA-256 sums, the
special token ids and the scheduler settings.

To check the result, the same pipeline the app runs (tokenize, one Euler step, decode) is run here with ONNX
Runtime on a few prompts, once with the full-precision and once with the quantized models, and the pictures are
saved in OUT/samples. The scheduler code below mirrors Diffusion.kt in the app.

    python prepare.py OUT
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

MODEL = "stabilityai/sd-turbo"
PROMPTS = [
    "a red lighthouse on a rocky coast at sunset, oil painting",
    "a cosy library full of books with a cat on an armchair",
    "a colourful hot air balloon festival over green hills",
]


def run(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---- the scheduler, as in Diffusion.kt ----

def sigmas_for(cfg):
    n = cfg["num_train_timesteps"]
    betas = np.linspace(cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5, n, dtype=np.float64) ** 2
    alphas_cumprod = np.cumprod(1.0 - betas)
    return np.sqrt((1 - alphas_cumprod) / alphas_cumprod)


def trailing_timesteps(n_train, steps):
    return [int(round(n_train - i * n_train / steps)) - 1 for i in range(steps)]


def generate(sessions, tokenizer, cfg, prompt, seed, steps=1):
    text, unet, vae = sessions
    ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True).input_ids
    id_type = np.int64 if text.get_inputs()[0].type == "tensor(int64)" else np.int32
    hidden = text.run(None, {text.get_inputs()[0].name: np.array([ids], dtype=id_type)})[0].astype(np.float32)

    sigmas = sigmas_for(cfg)
    timesteps = trailing_timesteps(cfg["num_train_timesteps"], steps)
    sig = [float(sigmas[t]) for t in timesteps] + [0.0]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 4, 64, 64)) * sigmas.max()

    names = {i.name: i for i in unet.get_inputs()}
    t_type = np.float32 if names["timestep"].type == "tensor(float)" else np.int64
    for i, t in enumerate(timesteps):
        s = sig[i]
        model_in = (x / np.sqrt(s * s + 1)).astype(np.float32)
        out = unet.run(None, {
            "sample": model_in,
            "timestep": np.array([t], dtype=t_type),
            "encoder_hidden_states": hidden,
        })[0].astype(np.float64)
        if cfg["prediction_type"] == "epsilon":
            x0 = x - s * out
        else:  # v_prediction
            x0 = out * (-s / np.sqrt(s * s + 1)) + x / (s * s + 1)
        x = x + (x - x0) / s * (sig[i + 1] - s)

    img = vae.run(None, {vae.get_inputs()[0].name: (x / 0.18215).astype(np.float32)})[0][0]
    img = np.clip(img / 2 + 0.5, 0, 1)
    return Image.fromarray((img.transpose(1, 2, 0) * 255).round().astype(np.uint8)), ids


def main():
    out = os.path.abspath(sys.argv[1])
    work = os.path.join(os.path.dirname(out), "onnx-fp32")
    os.makedirs(out, exist_ok=True)

    run("optimum-cli", "export", "onnx", "--model", MODEL, work)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    for part in ["text_encoder", "unet"]:
        t = time.time()
        quantize_dynamic(
            model_input=os.path.join(work, part, "model.onnx"),
            model_output=os.path.join(out, part + ".onnx"),
            weight_type=QuantType.QUInt8,  # ConvInteger only takes unsigned weights
            op_types_to_quantize=["MatMul", "Gemm", "Conv"],
            use_external_data_format=False,
        )
        print(f"quantized {part} in {time.time() - t:.0f} s", flush=True)
    # The decoder is small and loses visible quality when quantized, so it stays as it is.
    shutil.copy(os.path.join(work, "vae_decoder", "model.onnx"), os.path.join(out, "vae_decoder.onnx"))
    for name in ["vocab.json", "merges.txt"]:
        shutil.copy(os.path.join(work, "tokenizer", name), os.path.join(out, name))
    urllib.request.urlretrieve(f"https://huggingface.co/{MODEL}/raw/main/LICENSE.md", os.path.join(out, "LICENSE.md"))

    with open(os.path.join(work, "scheduler", "scheduler_config.json")) as f:
        sched = json.load(f)
    assert sched["_class_name"] == "EulerDiscreteScheduler", sched
    assert sched["timestep_spacing"] == "trailing", sched
    assert sched["beta_schedule"] == "scaled_linear", sched

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(work, "tokenizer"))

    # ---- check: the app's pipeline with full and reduced precision ----
    import onnxruntime as ort

    samples = os.path.join(os.path.dirname(out), "samples")
    os.makedirs(samples, exist_ok=True)
    checks = []
    for label, paths in [
        ("fp32", [os.path.join(work, p, "model.onnx") for p in ["text_encoder", "unet", "vae_decoder"]]),
        ("int8", [os.path.join(out, p + ".onnx") for p in ["text_encoder", "unet", "vae_decoder"]]),
    ]:
        sessions = [ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in paths]
        if label == "int8":
            print("unet inputs:", [(i.name, i.type, i.shape) for i in sessions[1].get_inputs()])
            print("text inputs:", [(i.name, i.type, i.shape) for i in sessions[0].get_inputs()])
            print("vae inputs:", [(i.name, i.type, i.shape) for i in sessions[2].get_inputs()])
        for k, prompt in enumerate(PROMPTS):
            for steps in [1, 2]:
                t = time.time()
                img, ids = generate(sessions, tokenizer, sched, prompt, seed=k, steps=steps)
                print(f"{label} steps={steps} '{prompt}': {time.time() - t:.1f} s", flush=True)
                img.save(os.path.join(samples, f"{label}-{k}-steps{steps}.png"))
                if label == "int8" and steps == 1:
                    checks.append({"prompt": prompt, "ids": ids})
        del sessions

    files = []
    for name in sorted(os.listdir(out)):
        path = os.path.join(out, name)
        files.append({"name": name, "size": os.path.getsize(path), "sha256": sha256(path)})
    manifest = {
        "version": 1,
        "model": MODEL,
        "files": files,
        "tokenizer": {
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
            "max_length": 77,
            "checks": checks,
        },
        "scheduler": {k: sched[k] for k in ["num_train_timesteps", "beta_start", "beta_end", "prediction_type"]},
        "vae_scaling_factor": 0.18215,
        "latent_size": 64,
    }
    with open(os.path.join(out, "model.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    for entry in files:
        print(f"{entry['name']}: {entry['size'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
