import json
import shutil
from pathlib import Path
from enum import Enum

import typer
from tqdm import tqdm

from ..utils import (
    alnum_to_asr_tokens,
    ExtendedPath,
    asr_manifest_reader,
    asr_manifest_writer,
    tscript_uuid_fname,
    get_mongo_conn,
    plot_seg,
)

app = typer.Typer()


def preprocess_datapoint(
    idx, rel_root, sample, use_domain_asr, annotation_only, enable_plots
):
    from pydub import AudioSegment
    from nemo.collections.asr.metrics import word_error_rate
    from jasper.client import transcribe_gen

    try:
        res = dict(sample)
        res["real_idx"] = idx
        audio_path = rel_root / Path(sample["audio_filepath"])
        res["audio_path"] = str(audio_path)
        if use_domain_asr:
            res["spoken"] = alnum_to_asr_tokens(res["text"])
        else:
            res["spoken"] = res["text"]
        res["utterance_id"] = audio_path.stem
        if not annotation_only:
            transcriber_pretrained = transcribe_gen(asr_port=8044)

            aud_seg = (
                AudioSegment.from_file_using_temporary_files(audio_path)
                .set_channels(1)
                .set_sample_width(2)
                .set_frame_rate(24000)
            )
            res["pretrained_asr"] = transcriber_pretrained(aud_seg.raw_data)
            res["pretrained_wer"] = word_error_rate(
                [res["text"]], [res["pretrained_asr"]]
            )
            if use_domain_asr:
                transcriber_speller = transcribe_gen(asr_port=8045)
                res["domain_asr"] = transcriber_speller(aud_seg.raw_data)
                res["domain_wer"] = word_error_rate(
                    [res["spoken"]], [res["pretrained_asr"]]
                )
        if enable_plots:
            wav_plot_path = (
                rel_root / Path("wav_plots") / Path(audio_path.name).with_suffix(".png")
            )
            if not wav_plot_path.exists():
                plot_seg(wav_plot_path, audio_path)
            res["plot_path"] = str(wav_plot_path)
        return res
    except BaseException as e:
        print(f'failed on {idx}: {sample["audio_filepath"]} with {e}')


@app.command()
def dump_ui(
    data_name: str = typer.Option("dataname", show_default=True),
    dataset_dir: Path = Path("./data/asr_data"),
    dump_dir: Path = Path("./data/valiation_data"),
    dump_fname: Path = typer.Option(Path("ui_dump.json"), show_default=True),
    use_domain_asr: bool = False,
    annotation_only: bool = False,
    enable_plots: bool = True,
):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    data_manifest_path = dataset_dir / Path(data_name) / Path("manifest.json")
    dump_path: Path = dump_dir / Path(data_name) / dump_fname
    plot_dir = data_manifest_path.parent / Path("wav_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Using data manifest:{data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        data_jsonl = pf.readlines()
        data_funcs = [
            partial(
                preprocess_datapoint,
                i,
                data_manifest_path.parent,
                json.loads(v),
                use_domain_asr,
                annotation_only,
                enable_plots,
            )
            for i, v in enumerate(data_jsonl)
        ]

        def exec_func(f):
            return f()

        with ThreadPoolExecutor() as exe:
            print("starting all preprocess tasks")
            data_final = filter(
                None,
                list(
                    tqdm(
                        exe.map(exec_func, data_funcs),
                        position=0,
                        leave=True,
                        total=len(data_funcs),
                    )
                ),
            )
    if annotation_only:
        result = list(data_final)
    else:
        wer_key = "domain_wer" if use_domain_asr else "pretrained_wer"
        result = sorted(data_final, key=lambda x: x[wer_key], reverse=True)
    ui_config = {
        "use_domain_asr": use_domain_asr,
        "annotation_only": annotation_only,
        "enable_plots": enable_plots,
        "data": result,
    }
    ExtendedPath(dump_path).write_json(ui_config)


@app.command()
def sample_ui(
    data_name: str = typer.Option("dataname", show_default=True),
    dump_dir: Path = Path("./data/asr_data"),
    dump_file: Path = Path("ui_dump.json"),
    sample_count: int = typer.Option(80, show_default=True),
    sample_file: Path = Path("sample_dump.json"),
):
    import pandas as pd

    processed_data_path = dump_dir / Path(data_name) / dump_file
    sample_path = dump_dir / Path(data_name) / sample_file
    processed_data = ExtendedPath(processed_data_path).read_json()
    df = pd.DataFrame(processed_data["data"])
    samples_per_caller = sample_count // len(df["caller"].unique())
    caller_samples = pd.concat(
        [g.sample(samples_per_caller) for (c, g) in df.groupby("caller")]
    )
    caller_samples = caller_samples.reset_index(drop=True)
    caller_samples["real_idx"] = caller_samples.index
    sample_data = caller_samples.to_dict("records")
    processed_data["data"] = sample_data
    typer.echo(f"sampling {sample_count} datapoints")
    ExtendedPath(sample_path).write_json(processed_data)


@app.command()
def task_ui(
    data_name: str = typer.Option("dataname", show_default=True),
    dump_dir: Path = Path("./data/asr_data"),
    dump_file: Path = Path("ui_dump.json"),
    task_count: int = typer.Option(4, show_default=True),
    task_file: str = "task_dump",
):
    import pandas as pd
    import numpy as np

    processed_data_path = dump_dir / Path(data_name) / dump_file
    processed_data = ExtendedPath(processed_data_path).read_json()
    df = pd.DataFrame(processed_data["data"]).sample(frac=1).reset_index(drop=True)
    for t_idx, task_f in enumerate(np.array_split(df, task_count)):
        task_f = task_f.reset_index(drop=True)
        task_f["real_idx"] = task_f.index
        task_data = task_f.to_dict("records")
        processed_data["data"] = task_data
        task_path = dump_dir / Path(data_name) / Path(task_file + f"-{t_idx}.json")
        ExtendedPath(task_path).write_json(processed_data)


@app.command()
def dump_corrections(
    task_uid: str,
    data_name: str = typer.Option("dataname", show_default=True),
    dump_dir: Path = Path("./data/asr_data"),
    dump_fname: Path = Path("corrections.json"),
):
    dump_path = dump_dir / Path(data_name) / dump_fname
    col = get_mongo_conn(col="asr_validation")
    task_id = [c for c in col.distinct("task_id") if c.rsplit("-", 1)[1] == task_uid][0]
    corrections = list(col.find({"type": "correction"}, projection={"_id": False}))
    cursor_obj = col.find({"type": "correction", "task_id": task_id}, projection={"_id": False})
    corrections = [c for c in cursor_obj]
    ExtendedPath(dump_path).write_json(corrections)


@app.command()
def caller_quality(
    task_uid: str,
    data_name: str = typer.Option("dataname", show_default=True),
    dump_dir: Path = Path("./data/asr_data"),
    dump_fname: Path = Path("ui_dump.json"),
    correction_fname: Path = Path("corrections.json"),
):
    import copy
    import pandas as pd

    dump_path = dump_dir / Path(data_name) / dump_fname
    correction_path = dump_dir / Path(data_name) / correction_fname
    dump_data = ExtendedPath(dump_path).read_json()

    dump_map = {d["utterance_id"]: d for d in dump_data["data"]}
    correction_data = ExtendedPath(correction_path).read_json()

    def correction_dp(c):
        dp = copy.deepcopy(dump_map[c["code"]])
        dp["valid"] = c["value"]["status"] == "Correct"
        return dp

    corrected_dump = [
        correction_dp(c)
        for c in correction_data
        if c["task_id"].rsplit("-", 1)[1] == task_uid
    ]
    df = pd.DataFrame(corrected_dump)
    print(f"Total samples: {len(df)}")
    for (c, g) in df.groupby("caller"):
        total = len(g)
        valid = len(g[g["valid"] == True])
        valid_rate = valid * 100 / total
        print(f"Caller: {c} Valid%:{valid_rate:.2f} of {total} samples")


@app.command()
def fill_unannotated(
    data_name: str = typer.Option("dataname", show_default=True),
    dump_dir: Path = Path("./data/valiation_data"),
    dump_file: Path = Path("ui_dump.json"),
    corrections_file: Path = Path("corrections.json"),
):
    processed_data_path = dump_dir / Path(data_name) / dump_file
    corrections_path = dump_dir / Path(data_name) / corrections_file
    processed_data = json.load(processed_data_path.open())
    corrections = json.load(corrections_path.open())
    annotated_codes = {c["code"] for c in corrections}
    all_codes = {c["gold_chars"] for c in processed_data}
    unann_codes = all_codes - annotated_codes
    mongo_conn = get_mongo_conn(col="asr_validation")
    for c in unann_codes:
        mongo_conn.find_one_and_update(
            {"type": "correction", "code": c},
            {"$set": {"value": {"status": "Inaudible", "correction": ""}}},
            upsert=True,
        )


@app.command()
def split_extract(
    data_name: str = typer.Option("dataname", show_default=True),
    # dest_data_name: str = typer.Option("call_aldata_namephanum_date", show_default=True),
    # dump_dir: Path = Path("./data/valiation_data"),
    dump_dir: Path = Path("./data/asr_data"),
    dump_file: Path = Path("ui_dump.json"),
    manifest_file: Path = Path("manifest.json"),
    corrections_file: str = typer.Option("corrections.json", show_default=True),
    conv_data_path: Path = typer.Option(Path("./data/conv_data.json"), show_default=True),
    extraction_type: str = "all",
):
    import shutil

    data_manifest_path = dump_dir / Path(data_name) / manifest_file
    conv_data = ExtendedPath(conv_data_path).read_json()

    def extract_data_of_type(extraction_key):
        extraction_vals = conv_data[extraction_key]
        dest_data_name = data_name + "_" + extraction_key.lower()

        manifest_gen = asr_manifest_reader(data_manifest_path)
        dest_data_dir = dump_dir / Path(dest_data_name)
        dest_data_dir.mkdir(exist_ok=True, parents=True)
        (dest_data_dir / Path("wav")).mkdir(exist_ok=True, parents=True)
        dest_manifest_path = dest_data_dir / manifest_file
        dest_ui_path = dest_data_dir / dump_file

        def extract_manifest(mg):
            for m in mg:
                if m["text"] in extraction_vals:
                    shutil.copy(m["audio_path"], dest_data_dir / Path(m["audio_filepath"]))
                    yield m

        asr_manifest_writer(dest_manifest_path, extract_manifest(manifest_gen))

        ui_data_path = dump_dir / Path(data_name) / dump_file
        orig_ui_data = ExtendedPath(ui_data_path).read_json()
        ui_data = orig_ui_data["data"]
        file_ui_map = {Path(u["audio_filepath"]).stem: u for u in ui_data}
        extracted_ui_data = list(filter(lambda u: u["text"] in extraction_vals, ui_data))
        final_data = []
        for i, d in enumerate(extracted_ui_data):
            d['real_idx'] = i
            final_data.append(d)
        orig_ui_data['data'] = final_data
        ExtendedPath(dest_ui_path).write_json(orig_ui_data)

        if corrections_file:
            dest_correction_path = dest_data_dir / corrections_file
            corrections_path = dump_dir / Path(data_name) / corrections_file
            corrections = json.load(corrections_path.open())
            extracted_corrections = list(
                filter(
                    lambda c: c["code"] in file_ui_map
                    and file_ui_map[c["code"]]["text"] in extraction_vals,
                    corrections,
                )
            )
            ExtendedPath(dest_correction_path).write_json(extracted_corrections)

    if extraction_type.value == 'all':
        for ext_key in conv_data.keys():
            extract_data_of_type(ext_key)
    else:
        extract_data_of_type(extraction_type.value)


@app.command()
def update_corrections(
    data_name: str = typer.Option("dataname", show_default=True),
    dump_dir: Path = Path("./data/asr_data"),
    manifest_file: Path = Path("manifest.json"),
    corrections_file: Path = Path("corrections.json"),
    ui_dump_file: Path = Path("ui_dump.json"),
    skip_incorrect: bool = typer.Option(True, show_default=True),
):
    data_manifest_path = dump_dir / Path(data_name) / manifest_file
    corrections_path = dump_dir / Path(data_name) / corrections_file
    ui_dump_path = dump_dir / Path(data_name) / ui_dump_file

    def correct_manifest(ui_dump_path, corrections_path):
        corrections = ExtendedPath(corrections_path).read_json()
        ui_data = ExtendedPath(ui_dump_path).read_json()['data']
        correct_set = {
            c["code"] for c in corrections if c["value"]["status"] == "Correct"
        }
        # incorrect_set = {c["code"] for c in corrections if c["value"]["status"] == "Inaudible"}
        correction_map = {
            c["code"]: c["value"]["correction"]
            for c in corrections
            if c["value"]["status"] == "Incorrect"
        }
        # for d in manifest_data_gen:
        #     if d["chars"] in incorrect_set:
        #         d["audio_path"].unlink()
        # renamed_set = set()
        for d in ui_data:
            if d["utterance_id"] in correct_set:
                yield {
                    "audio_filepath": d["audio_filepath"],
                    "duration": d["duration"],
                    "text": d["text"],
                }
            elif d["utterance_id"] in correction_map:
                correct_text = correction_map[d["utterance_id"]]
                if skip_incorrect:
                    print(
                        f'skipping incorrect {d["audio_path"]} corrected to {correct_text}'
                    )
                else:
                    orig_audio_path = Path(d["audio_path"])
                    new_name = str(Path(tscript_uuid_fname(correct_text)).with_suffix(".wav"))
                    new_audio_path = orig_audio_path.with_name(new_name)
                    orig_audio_path.replace(new_audio_path)
                    new_filepath = str(Path(d["audio_filepath"]).with_name(new_name))
                    yield {
                        "audio_filepath": new_filepath,
                        "duration": d["duration"],
                        "text": correct_text,
                    }
            else:
                orig_audio_path = Path(d["audio_path"])
                # don't delete if another correction points to an old file
                # if d["text"] not in renamed_set:
                orig_audio_path.unlink()
                # else:
                #     print(f'skipping deletion of correction:{d["text"]}')

    typer.echo(f"Using data manifest:{data_manifest_path}")
    dataset_dir = data_manifest_path.parent
    dataset_name = dataset_dir.name
    backup_dir = dataset_dir.with_name(dataset_name + ".bkp")
    if not backup_dir.exists():
        typer.echo(f"backing up to :{backup_dir}")
        shutil.copytree(str(dataset_dir), str(backup_dir))
    # manifest_gen = asr_manifest_reader(data_manifest_path)
    corrected_manifest = correct_manifest(ui_dump_path, corrections_path)
    new_data_manifest_path = data_manifest_path.with_name("manifest.new")
    asr_manifest_writer(new_data_manifest_path, corrected_manifest)
    new_data_manifest_path.replace(data_manifest_path)


@app.command()
def clear_mongo_corrections():
    delete = typer.confirm("are you sure you want to clear mongo collection it?")
    if delete:
        col = get_mongo_conn(col="asr_validation")
        col.delete_many({"type": "correction"})
        col.delete_many({"type": "current_cursor"})
        typer.echo("deleted mongo collection.")
        return
    typer.echo("Aborted")


def main():
    app()


if __name__ == "__main__":
    main()
