A short example script to run the smallest size of Google Gemma3 large language model (270M parameters).

- [Ubuntu](#ubuntu)
  - [1. Setup](#1-setup)
  - [2. Run script](#2-run-script)
- [macOS](#macos)
  - [1. Setup](#1-setup-1)
  - [2. Run script](#2-run-script-1)
- [References](#references)


# Ubuntu

These steps tested on:

> * Ubuntu 24.04.02 LTS
> * Python 3.12.3
> * NVidia RTX A2000 GPU (Ampere)
> * CUDA 12.8

## 1. Setup

Get an access token from your HuggingFace account and paste it to this command.
```bash
HF_TOKEN=<your token>
```

Create a Python environment and activate it.
```bash
sudo apt update
sudo apt upgrade
sudo apt install python3-venv

cd
python3 -m venv venv_gemma3_270m
source ./venv_gemma3_270m/bin/activate
```

Install package requirements.
```bash
python3 -m pip install --upgrade pip
cd gemma3_270m
python3 -m pip install -r requirements.txt
```

## 2. Run script

```bash
python3 main.py --hf_token=${HF_TOKEN}
```

Example run.
> The generated text will change each run: the model decoding is stochastic.
```console
$ python main.py --hf_token=${HF_TOKEN}
Device set to use cuda:0
Model:        google/gemma-3-270m-it
Precision:    torch.bfloat16
================================================================================
Input prompt: What causes climate change?
Climate change is caused by human activities that release greenhouse gases into the atmosphere. These gases trap heat and warm the planet.
```

# macOS

These steps tested on:

> * MacBook Air M3 16GB
> * macOS 15.6.1
> * Python 3.12.6
> * PyTorch 2.8.0

## 1. Setup

Get an access token from your HuggingFace account and paste it to this command.
```bash
HF_TOKEN=<your token>
```

Create a Python environment and activate it.
```bash
brew install venv

cd
python3 -m venv .venv_gemma3
source ./.venv_gemma3/bin/activate
```

Install package requirements.
```bash
python3 -m pip install --upgrade pip
cd gemma3_270m
python3 -m pip install -r requirements.txt
```

## 2. Run script

```bash
python3 main.py --hf_token=${HF_TOKEN}
```

Example run.
> The generated text will change each run: the model decoding is stochastic.
```console
$ python3 main.py --hf_token=${HF_TOKEN}
Device set to use mps
Model:        google/gemma-3-270m-it
Device:       mps:0
Precision:    torch.bfloat16
================================================================================
Input prompt: What causes climate change?
Climate change is caused by human activities, primarily the burning of fossil fuels.
```

# References

* https://huggingface.co/
* https://huggingface.co/google/gemma-3-270m-it
* https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune
