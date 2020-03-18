import os
import logging

import rpyc
from rpyc.utils.server import ThreadedServer

from .asr import JasperASR


MODEL_YAML = os.environ.get("JASPER_MODEL_CONFIG", "/models/jasper/jasper10x5dr.yaml")
CHECKPOINT_ENCODER = os.environ.get(
    "JASPER_ENCODER_CHECKPOINT", "/models/jasper/JasperEncoder-STEP-265520.pt"
)
CHECKPOINT_DECODER = os.environ.get(
    "JASPER_DECODER_CHECKPOINT", "/models/jasper/JasperDecoderForCTC-STEP-265520.pt"
)
KEN_LM = os.environ.get("JASPER_KEN_LM", None)

asr_recognizer = JasperASR(MODEL_YAML, CHECKPOINT_ENCODER, CHECKPOINT_DECODER, KEN_LM)


class ASRService(rpyc.Service):
    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_transcribe(self, utterance: bytes):  # this is an exposed method
        speech_audio = asr_recognizer.transcribe(utterance)
        return speech_audio

    def exposed_transcribe_cb(
        self, utterance: bytes, respond
    ):  # this is an exposed method
        speech_audio = asr_recognizer.transcribe(utterance)
        respond(speech_audio)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    port = int(os.environ.get("ASR_RPYC_PORT", "8044"))
    logging.info("starting tts server...")
    t = ThreadedServer(ASRService, port=port)
    t.start()


if __name__ == "__main__":
    main()
