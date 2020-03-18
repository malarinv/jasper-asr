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
KEN_LM = os.environ.get("JASPER_KEN_LM", "/models/jasper/kenlm.pt")


def arg_parser():
    prog = Path(__file__).stem
    parser = argparse.ArgumentParser(
        prog=prog, description=f"generates transcription of the audio_file"
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="audio file(16khz 1channel int16 wav) to transcribe",
    )
    parser.add_argument(
        "--greedy", type=bool, default=False, help="enables greedy decoding"
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


def main():
    parser = arg_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    audio_file = args_dict.pop("audio_file")
    greedy = args_dict.pop("greedy")
    jasper_asr = JasperASR(**args_dict)
    jasper_asr.transcribe_file(audio_file, greedy)
