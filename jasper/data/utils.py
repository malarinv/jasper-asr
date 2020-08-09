import io
import os
import json
import wave
from pathlib import Path
from functools import partial
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

import pymongo
from slugify import slugify
from jasper.client import transcribe_gen
from nemo.collections.asr.metrics import word_error_rate
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm


def manifest_str(path, dur, text):
    return (
        json.dumps({"audio_filepath": path, "duration": round(dur, 1), "text": text})
        + "\n"
    )


def wav_bytes(audio_bytes, frame_rate=24000):
    wf_b = io.BytesIO()
    with wave.open(wf_b, mode="w") as wf:
        wf.setnchannels(1)
        wf.setframerate(frame_rate)
        wf.setsampwidth(2)
        wf.writeframesraw(audio_bytes)
    return wf_b.getvalue()


def tscript_uuid_fname(transcript):
    return str(uuid4()) + "_" + slugify(transcript, max_length=8)


def asr_data_writer(output_dir, dataset_name, asr_data_source, verbose=False):
    dataset_dir = output_dir / Path(dataset_name)
    (dataset_dir / Path("wav")).mkdir(parents=True, exist_ok=True)
    asr_manifest = dataset_dir / Path("manifest.json")
    num_datapoints = 0
    with asr_manifest.open("w") as mf:
        print(f"writing manifest to {asr_manifest}")
        for transcript, audio_dur, wav_data in asr_data_source:
            fname = tscript_uuid_fname(transcript)
            audio_file = dataset_dir / Path("wav") / Path(fname).with_suffix(".wav")
            audio_file.write_bytes(wav_data)
            rel_data_path = audio_file.relative_to(dataset_dir)
            manifest = manifest_str(str(rel_data_path), audio_dur, transcript)
            mf.write(manifest)
            if verbose:
                print(f"writing '{transcript}' of duration {audio_dur}")
            num_datapoints += 1
    return num_datapoints


def ui_data_generator(output_dir, dataset_name, asr_data_source, verbose=False):
    dataset_dir = output_dir / Path(dataset_name)
    (dataset_dir / Path("wav")).mkdir(parents=True, exist_ok=True)
    (dataset_dir / Path("wav_plots")).mkdir(parents=True, exist_ok=True)

    def data_fn(
        transcript,
        audio_dur,
        wav_data,
        caller_name,
        aud_seg,
        fname,
        audio_path,
        num_datapoints,
        rel_data_path,
    ):
        pretrained_result = transcriber_pretrained(aud_seg.raw_data)
        pretrained_wer = word_error_rate([transcript], [pretrained_result])
        png_path = Path(fname).with_suffix(".png")
        wav_plot_path = dataset_dir / Path("wav_plots") / png_path
        if not wav_plot_path.exists():
            plot_seg(wav_plot_path, audio_path)
        return {
            "audio_filepath": str(rel_data_path),
            "duration": round(audio_dur, 1),
            "text": transcript,
            "real_idx": num_datapoints,
            "audio_path": audio_path,
            "spoken": transcript,
            "caller": caller_name,
            "utterance_id": fname,
            "pretrained_asr": pretrained_result,
            "pretrained_wer": pretrained_wer,
            "plot_path": str(wav_plot_path),
        }

    num_datapoints = 0
    data_funcs = []
    transcriber_pretrained = transcribe_gen(asr_port=8044)
    for transcript, audio_dur, wav_data, caller_name, aud_seg in asr_data_source:
        fname = str(uuid4()) + "_" + slugify(transcript, max_length=8)
        audio_file = dataset_dir / Path("wav") / Path(fname).with_suffix(".wav")
        audio_file.write_bytes(wav_data)
        audio_path = str(audio_file)
        rel_data_path = audio_file.relative_to(dataset_dir)
        data_funcs.append(
            partial(
                data_fn,
                transcript,
                audio_dur,
                wav_data,
                caller_name,
                aud_seg,
                fname,
                audio_path,
                num_datapoints,
                rel_data_path,
            )
        )
        num_datapoints += 1
    ui_data = parallel_apply(lambda x: x(), data_funcs)
    return ui_data, num_datapoints


def ui_dump_manifest_writer(output_dir, dataset_name, asr_data_source, verbose=False):
    dataset_dir = output_dir / Path(dataset_name)
    dump_data, num_datapoints = ui_data_generator(
        output_dir, dataset_name, asr_data_source, verbose=verbose
    )

    asr_manifest = dataset_dir / Path("manifest.json")
    with asr_manifest.open("w") as mf:
        print(f"writing manifest to {asr_manifest}")
        for d in dump_data:
            rel_data_path = d["audio_filepath"]
            audio_dur = d["duration"]
            transcript = d["text"]
            manifest = manifest_str(str(rel_data_path), audio_dur, transcript)
            mf.write(manifest)

    ui_dump_file = dataset_dir / Path("ui_dump.json")
    ExtendedPath(ui_dump_file).write_json({"data": dump_data})
    return num_datapoints


def asr_manifest_reader(data_manifest_path: Path):
    print(f"reading manifest from {data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        data_jsonl = pf.readlines()
    data_data = [json.loads(v) for v in data_jsonl]
    for p in data_data:
        p["audio_path"] = data_manifest_path.parent / Path(p["audio_filepath"])
        p["text"] = p["text"].strip()
        yield p


def asr_manifest_writer(asr_manifest_path: Path, manifest_str_source):
    with asr_manifest_path.open("w") as mf:
        print(f"opening {asr_manifest_path} for writing manifest")
        for mani_dict in manifest_str_source:
            manifest = manifest_str(
                mani_dict["audio_filepath"], mani_dict["duration"], mani_dict["text"]
            )
            mf.write(manifest)


def asr_test_writer(out_file_path: Path, source):
    def dd_str(dd, idx):
        path = dd["audio_filepath"]
        # dur = dd["duration"]
        # return f"SAY {idx}\nPAUSE 3\nPLAY {path}\nPAUSE 3\n\n"
        return f"PAUSE 2\nPLAY {path}\nPAUSE 60\n\n"

    res_file = out_file_path.with_suffix(".result.json")
    with out_file_path.open("w") as of:
        print(f"opening {out_file_path} for writing test")
        results = []
        idx = 0
        for ui_dd in source:
            results.append(ui_dd)
            out_str = dd_str(ui_dd, idx)
            of.write(out_str)
            idx += 1
        of.write("DO_HANGUP\n")
        ExtendedPath(res_file).write_json(results)


def batch(iterable, n=1):
    ls = len(iterable)
    return [iterable[ndx : min(ndx + n, ls)] for ndx in range(0, ls, n)]


class ExtendedPath(type(Path())):
    """docstring for ExtendedPath."""

    def read_json(self):
        print(f"reading json from {self}")
        with self.open("r") as jf:
            return json.load(jf)

    def write_json(self, data):
        print(f"writing json to {self}")
        self.parent.mkdir(parents=True, exist_ok=True)
        with self.open("w") as jf:
            return json.dump(data, jf, indent=2)


def get_mongo_conn(host="", port=27017, db="test", col="calls"):
    mongo_host = host if host else os.environ.get("MONGO_HOST", "localhost")
    mongo_uri = f"mongodb://{mongo_host}:{port}/"
    return pymongo.MongoClient(mongo_uri)[db][col]


def strip_silence(sound):
    from pydub.silence import detect_leading_silence

    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    return sound[start_trim : duration - end_trim]


def plot_seg(wav_plot_path, audio_path):
    fig = plt.Figure()
    ax = fig.add_subplot()
    (y, sr) = librosa.load(audio_path)
    librosa.display.waveplot(y=y, sr=sr, ax=ax)
    with wav_plot_path.open("wb") as wav_plot_f:
        fig.set_tight_layout(True)
        fig.savefig(wav_plot_f, format="png", dpi=50)


def parallel_apply(fn, iterable, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as exe:
        print(f"parallelly applying {fn}")
        return [
            res
            for res in tqdm(
                exe.map(fn, iterable), position=0, leave=True, total=len(iterable)
            )
        ]
