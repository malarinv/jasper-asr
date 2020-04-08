import os
import tempfile
from ruamel.yaml import YAML
import json
import nemo
import nemo.collections.asr as nemo_asr
import wave
from nemo.collections.asr.helpers import post_process_predictions

logging = nemo.logging

WORK_DIR = "/tmp"


class JasperASR(object):
    """docstring for JasperASR."""

    def __init__(
        self, model_yaml, encoder_checkpoint, decoder_checkpoint, language_model=None
    ):
        super(JasperASR, self).__init__()
        # Read model YAML
        yaml = YAML(typ="safe")
        with open(model_yaml) as f:
            jasper_model_definition = yaml.load(f)
        self.neural_factory = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.GPU, backend=nemo.core.Backend.PyTorch
        )
        self.labels = jasper_model_definition["labels"]
        self.data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor()
        self.jasper_encoder = nemo_asr.JasperEncoder(
            jasper=jasper_model_definition["JasperEncoder"]["jasper"],
            activation=jasper_model_definition["JasperEncoder"]["activation"],
            feat_in=jasper_model_definition["AudioToMelSpectrogramPreprocessor"][
                "features"
            ],
        )
        self.jasper_encoder.restore_from(encoder_checkpoint, local_rank=0)
        self.jasper_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in=1024, num_classes=len(self.labels)
        )
        self.jasper_decoder.restore_from(decoder_checkpoint, local_rank=0)
        self.greedy_decoder = nemo_asr.GreedyCTCDecoder()
        self.beam_search_with_lm = None
        if language_model:
            self.beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
                vocab=self.labels,
                beam_width=64,
                alpha=2.0,
                beta=1.0,
                lm_path=language_model,
                num_cpus=max(os.cpu_count(), 1),
            )

    def transcribe(self, audio_data, greedy=True):
        audio_file = tempfile.NamedTemporaryFile(
            dir=WORK_DIR, prefix="jasper_audio.", delete=False
        )
        # audio_file.write(audio_data)
        audio_file.close()
        audio_file_path = audio_file.name
        wf = wave.open(audio_file_path, "w")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframesraw(audio_data)
        wf.close()
        manifest = {"audio_filepath": audio_file_path, "duration": 60, "text": "todo"}
        manifest_file = tempfile.NamedTemporaryFile(
            dir=WORK_DIR, prefix="jasper_manifest.", delete=False, mode="w"
        )
        manifest_file.write(json.dumps(manifest))
        manifest_file.close()
        manifest_file_path = manifest_file.name
        data_layer = nemo_asr.AudioToTextDataLayer(
            shuffle=False,
            manifest_filepath=manifest_file_path,
            labels=self.labels,
            batch_size=1,
        )

        # Define inference DAG
        audio_signal, audio_signal_len, _, _ = data_layer()
        processed_signal, processed_signal_len = self.data_preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        encoded, encoded_len = self.jasper_encoder(
            audio_signal=processed_signal, length=processed_signal_len
        )
        log_probs = self.jasper_decoder(encoder_output=encoded)
        predictions = self.greedy_decoder(log_probs=log_probs)

        if greedy:
            eval_tensors = [predictions]
        else:
            if self.beam_search_with_lm:
                logging.info("Running with beam search")
                beam_predictions = self.beam_search_with_lm(
                    log_probs=log_probs, log_probs_length=encoded_len
                )
                eval_tensors = [beam_predictions]
            else:
                logging.info(
                    "language_model not specified. falling back to greedy decoding."
                )
                eval_tensors = [predictions]

        tensors = self.neural_factory.infer(tensors=eval_tensors)
        prediction = post_process_predictions(tensors[0], self.labels)
        prediction_text = ". ".join(prediction)
        os.unlink(manifest_file.name)
        os.unlink(audio_file.name)
        return prediction_text

    def transcribe_file(self, audio_file, *args, **kwargs):
        tscript_file_path = audio_file.with_suffix(".txt")
        audio_file_path = str(audio_file)
        with wave.open(audio_file_path, "r") as af:
            frame_count = af.getnframes()
            audio_data = af.readframes(frame_count)
            transcription = self.transcribe(audio_data, *args, **kwargs)
            with open(tscript_file_path, "w") as tf:
                tf.write(transcription)
