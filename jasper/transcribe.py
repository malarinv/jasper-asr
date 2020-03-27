from pathlib import Path
from .asr import JasperASR
from .utils import arg_parser


def main():
    parser = arg_parser('jasper_transcribe')
    parser.description = 'transcribe audio file to text'
    parser.add_argument(
        "audio_file",
        type=Path,
        help="audio file(16khz 1channel int16 wav) to transcribe",
    )
    parser.add_argument(
        "--greedy", type=bool, default=False, help="enables greedy decoding"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    audio_file = args_dict.pop("audio_file")
    greedy = args_dict.pop("greedy")
    jasper_asr = JasperASR(**args_dict)
    jasper_asr.transcribe_file(audio_file, greedy)
