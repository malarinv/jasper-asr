# Jasper ASR

[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

> Generates text from speech audio
---

# Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)

# Features

* ASR using Jasper (from [NemoToolkit](https://github.com/NVIDIA/NeMo) )


# Installation
To install the packages and its dependencies run.
```bash
python setup.py install
```
or with pip
```bash
pip install .
```

The installation should work on Python 3.6 or newer. Untested on Python 2.7

# Usage
```python
from jasper.asr import JasperASR
asr_model = JasperASR("/path/to/model_config_yaml","/path/to/encoder_checkpoint","/path/to/decoder_checkpoint") # Loads the models
TEXT = asr_model.transcribe(wav_data) # Returns the text spoken in the wav
```
