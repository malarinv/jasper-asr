import os
import argparse
from pathlib import Path

MODEL_YAML = os.environ.get("JASPER_MODEL_CONFIG", "/models/jasper/jasper10x5dr.yaml")
CHECKPOINT_ENCODER = os.environ.get(
    "JASPER_ENCODER_CHECKPOINT", "/models/jasper/JasperEncoder-STEP-265520.pt"
)
CHECKPOINT_DECODER = os.environ.get(
    "JASPER_DECODER_CHECKPOINT", "/models/jasper/JasperDecoderForCTC-STEP-265520.pt"
)
KEN_LM = os.environ.get("JASPER_KEN_LM", "/models/jasper/kenlm.pt")


def arg_parser(prog):
    parser = argparse.ArgumentParser(
        prog=prog, description=f"convert speech to text"
    )
    parser.add_argument(
        "--model_yaml",
        type=Path,
        default=Path(MODEL_YAML),
        help="model config yaml file",
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=Path,
        default=Path(CHECKPOINT_ENCODER),
        help="encoder checkpoint weights file",
    )
    parser.add_argument(
        "--decoder_checkpoint",
        type=Path,
        default=Path(CHECKPOINT_DECODER),
        help="decoder checkpoint weights file",
    )
    parser.add_argument(
        "--language_model", type=Path, default=None, help="kenlm language model file"
    )
    return parser
