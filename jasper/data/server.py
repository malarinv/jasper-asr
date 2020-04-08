import os
from pathlib import Path

import typer
import rpyc
from rpyc.utils.server import ThreadedServer
import nemo
import pickle

# import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.segment import AudioSegment

app = typer.Typer()

nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch, placement=nemo.core.DeviceType.CPU
)


class ASRDataService(rpyc.Service):
    def exposed_get_path_samples(
        self, file_path, target_sr, int_values, offset, duration, trim
    ):
        print(f"loading.. {file_path}")
        audio = AudioSegment.from_file(
            file_path,
            target_sr=target_sr,
            int_values=int_values,
            offset=offset,
            duration=duration,
            trim=trim,
        )
        # print(f"returning.. {len(audio.samples)} items of type{type(audio.samples)}")
        return pickle.dumps(audio.samples)

    def exposed_read_path(self, file_path):
        # print(f"reading path.. {file_path}")
        return Path(file_path).read_bytes()


@app.command()
def run_server(port: int = 0):
    listen_port = port if port else int(os.environ.get("ASR_DARA_RPYC_PORT", "8064"))
    service = ASRDataService()
    t = ThreadedServer(
        service, port=listen_port, protocol_config={"allow_all_attrs": True}
    )
    typer.echo(f"starting asr server on {listen_port}...")
    t.start()


def main():
    app()


if __name__ == "__main__":
    main()
