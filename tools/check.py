"""
Checks the published model: downloads it from the release, runs the app's pipeline (as in prepare.py) on a few
prompts with 1 and 2 steps, and saves the pictures and timings in OUT (samples-*.png, timings.txt).

    python check.py MODEL_DIR OUT
"""
import json
import os
import sys
import time

import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from prepare import PROMPTS, generate  # noqa: E402


def main():
    model, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(model, "model.json")) as f:
        manifest = json.load(f)
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer(os.path.join(model, "vocab.json"), os.path.join(model, "merges.txt"), pad_token="!")
    lines = []
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # as on the phone
    t = time.time()
    sessions = [ort.InferenceSession(os.path.join(model, p + ".onnx"), opts, providers=["CPUExecutionProvider"])
                for p in ["text_encoder", "unet", "vae_decoder"]]
    lines.append(f"load: {time.time() - t:.1f} s")
    for k, prompt in enumerate(PROMPTS):
        for steps in [1, 2]:
            t = time.time()
            img, _ = generate(sessions, tokenizer, manifest["scheduler"], prompt, seed=k, steps=steps)
            lines.append(f"steps={steps} '{prompt}': {time.time() - t:.1f} s")
            img.save(os.path.join(out, f"sample-{k}-steps{steps}.png"))
    with open(os.path.join(out, "timings.txt"), "w") as f:
        f.write("Runner CPU, 4 threads\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
