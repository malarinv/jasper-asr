from rastrik.proto.callrecord_pb2 import CallRecordEvent, CallRecord
import gzip
from pydub import AudioSegment
import json
from .utils import ExtendedPath, asr_data_writer, strip_silence

import typer
from itertools import chain
from io import BytesIO
from pathlib import Path

app = typer.Typer()

@app.command()
def extract_manifest(
    call_audio_dir: Path = Path("./data/call_audio"),
    call_meta_dir: Path = Path("./data/call_metadata"),
    output_dir: Path = Path("./data"),
    dataset_name: str = "grassroot_pizzahut_v1",
    verbose: bool = False,
):


    call_asr_data: Path = output_dir / Path("asr_data")
    call_asr_data.mkdir(exist_ok=True, parents=True)
    """
    def read_event_old(log_file,audio_file):
        call_wav = AudioSegment.from_wav(audio_file)
        call_wav_0, call_wav_1 = call_wav.split_to_mono()
        with gzip.open(log_file, "rb") as log_h:
            record_data = log_h.read()
        cr = CallRecord()
        cr.ParseFromString(record_data)

        import pdb
        first_audio_event_timestamp = next ((i
            for i in cr.events
                if i.WhichOneof("event_type") == "call_event"
                and i.call_event.WhichOneof("event_type") == "call_audio"
        )).timestamp.ToDatetime()

        speech_events = [ i
            for i in cr.events
            if i.WhichOneof("event_type") == "asr_result"
        ]
        previous_event_timestamp = first_audio_event_timestamp - first_audio_event_timestamp
        for index,each_speech_events in enumerate(speech_events):
            asr_final = each_speech_events.asr_result.text
            speech_timestamp = each_speech_events.timestamp.ToDatetime()
            actual_timestamp = speech_timestamp - first_audio_event_timestamp
            print(previous_event_timestamp.total_seconds(),actual_timestamp.total_seconds(),asr_final)
            start_time = previous_event_timestamp.total_seconds()*1000
            end_time = actual_timestamp.total_seconds() * 1000
            audio_segment = strip_silence(call_wav_1[start_time:end_time])
            audio_segment.export(output_folder+str(index) + '.wav' ,format='wav')
            previous_event_timestamp = actual_timestamp
    """

    def wav_pb2_generator(call_audio_dir):
        for wav_path in call_audio_dir.glob("**/*.wav"):
            if verbose:
                typer.echo(f"loading events for file {wav_path}")
            call_wav = AudioSegment.from_file_using_temporary_files(wav_path)
            rel_meta_path = wav_path.with_suffix(".pb2.gz").relative_to(call_audio_dir)
            meta_path = call_meta_dir / rel_meta_path
            #events = ExtendedPath(meta_path).read_json()
            yield  call_wav,wav_path, meta_path

    def read_event(call_wav,log_file):
        #call_wav = AudioSegment.from_wav(audio_file)
        call_wav_0, call_wav_1 = call_wav.split_to_mono()
        with gzip.open(log_file, "rb") as log_h:
            record_data = log_h.read()
        cr = CallRecord()
        cr.ParseFromString(record_data)

        import pdb
        first_audio_event_timestamp = next ((i
            for i in cr.events
                if i.WhichOneof("event_type") == "call_event"
                and i.call_event.WhichOneof("event_type") == "call_audio"
        )).timestamp.ToDatetime()

        speech_events = [ i
            for i in cr.events
            if i.WhichOneof("event_type") == "speech_event"
            and i.speech_event.WhichOneof("event_type") == "asr_final"
        ]
        previous_event_timestamp = first_audio_event_timestamp - first_audio_event_timestamp
        for index,each_speech_events in enumerate(speech_events):
            asr_final = each_speech_events.speech_event.asr_final
            speech_timestamp = each_speech_events.timestamp.ToDatetime()
            actual_timestamp = speech_timestamp - first_audio_event_timestamp
            print(previous_event_timestamp.total_seconds(),actual_timestamp.total_seconds(),asr_final)
            start_time = previous_event_timestamp.total_seconds()*1000
            end_time = actual_timestamp.total_seconds() * 1000
            audio_segment = strip_silence(call_wav_1[start_time:end_time])

            code_fb = BytesIO()
            audio_segment.export(code_fb, format="wav")
            wav_data = code_fb.getvalue()

            #output_audio_path = output_folder + audio_file.replace('.wav','') + '_' + str(index) 
            #audio_segment.export( output_audio_path+ '.wav' ,format='wav')
            #manifest_file.write(json.dumps({"audio_filepath":output_audio_path , "duration": (end_time-start_time) / 1000 , "text":asr_final }) + '\n')
            previous_event_timestamp = actual_timestamp
            duration = (end_time-start_time) / 1000
            yield asr_final,duration,wav_data


    def generate_call_asr_data():
        full_asr_data = []
        total_duration = 0
        for wav,wav_path, pb2_path in wav_pb2_generator(call_audio_dir):
            asr_data = read_event(wav,pb2_path)
            total_duration += wav.duration_seconds
            full_asr_data.append(asr_data)

        typer.echo(f"loaded {len(full_asr_data)} calls of duration {total_duration}s")
        n_dps = asr_data_writer(call_asr_data, dataset_name, chain(*full_asr_data))
        typer.echo(f"written {n_dps} data points")


    generate_call_asr_data()

def main():
    app()


if __name__ == "__main__":
    main()
    





