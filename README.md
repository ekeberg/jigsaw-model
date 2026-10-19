# Jigsaw image model

The on-device image generation model used by the Jigsaw puzzle app for Android. The app downloads the files from
this repository's releases when image generation is turned on in its settings. This repository is public so the
app can fetch the files without signing in; the app itself lives elsewhere.

## The model

[SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) by Stability AI, exported to ONNX:

| File | What it is |
| --- | --- |
| `text_encoder.onnx` | CLIP text encoder, weights quantized to 8 bits |
| `unet.onnx` | the denoising UNet, weights quantized to 8 bits |
| `vae_decoder.onnx` | the VAE decoder, full precision (it loses visible quality when quantized) |
| `vocab.json`, `merges.txt` | the CLIP tokenizer |
| `model.json` | manifest the app reads: file sizes and SHA-256 sums, special token ids, reference token ids for a few prompts, scheduler settings |
| `LICENSE.md` | the model's license |

The release also carries sample pictures generated from the published files, and timings (`timings.txt`).

SD-Turbo is licensed under the Stability AI Community License (see `LICENSE.md` in the release). Powered by
Stability AI.

## Building it

`.github/workflows/model.yml` runs `tools/prepare.py`. That script exports the model with Optimum and quantizes
it with ONNX Runtime. It then runs the app's pipeline (tokenize, Euler steps, decode) on a few prompts at full and
reduced precision as a check, and writes the manifest. The workflow publishes the result as a release. A second job
downloads the published files, generates samples with `tools/check.py` and attaches them to the release.

To publish a new model version, change `MODEL_TAG` in the workflow and `TAG` in the app's `ImageModel.kt` together.
Phones holding the old version then download the new one. The workflow skips the build when a release with that tag
already exists.
