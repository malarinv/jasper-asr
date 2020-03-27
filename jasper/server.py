import os
import logging

import rpyc
from rpyc.utils.server import ThreadedServer

from .asr import JasperASR
from .utils import arg_parser


class ASRService(rpyc.Service):
    def __init__(self, asr_recognizer):
        self.asr = asr_recognizer

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_transcribe(self, utterance: bytes):  # this is an exposed method
        speech_audio = self.asr.transcribe(utterance)
        return speech_audio

    def exposed_transcribe_cb(
        self, utterance: bytes, respond
    ):  # this is an exposed method
        speech_audio = self.asr.transcribe(utterance)
        respond(speech_audio)


def main():
    parser = arg_parser('jasper_transcribe')
    parser.description = 'jasper asr rpyc server'
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ASR_RPYC_PORT", "8044")), help="port to listen on"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    port = args_dict.pop("port")
    jasper_asr = JasperASR(**args_dict)
    service = ASRService(jasper_asr)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("starting asr server...")
    t = ThreadedServer(service, port=port)
    t.start()


if __name__ == "__main__":
    main()
