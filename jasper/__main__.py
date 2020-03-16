import os
import argparse
from pathlib import Path
from .asr import JasperASR

MODEL_YAML = os.environ.get("JASPER_MODEL_CONFIG", "/models/jasper/jasper10x5dr.yaml")
CHECKPOINT_ENCODER = os.environ.get(
    "JASPER_ENCODER_CHECKPOINT", "/models/jasper/JasperEncoder-STEP-265520.pt"
)
CHECKPOINT_DECODER = os.environ.get(
    "JASPER_DECODER_CHECKPOINT", "/models/jasper/JasperDecoderForCTC-STEP-265520.pt"
)


def arg_parser():
    prog = Path(__file__).stem
    parser = argparse.ArgumentParser(
        prog=prog, description=f"generates transcription of the audio_file"
    )
    parser.add_argument(
        "--audio_file",
        type=Path,
        help="audio file(16khz 1channel int16 wav) to transcribe",
    )
    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()
    jasper_asr = JasperASR(MODEL_YAML, CHECKPOINT_ENCODER, CHECKPOINT_DECODER)
    jasper_asr.transcribe_file(args.audio_file)
