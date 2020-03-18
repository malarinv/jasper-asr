from setuptools import setup

requirements = [
    "ruamel.yaml",
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@09e3ba4dfe333f86d6c5c1048e07210924294be9#egg=nemo_toolkit",
]

extra_requirements = {"server": ["rpyc==4.1.4"]}

setup(
    name="jasper-asr",
    version="0.1",
    description="Tool to get gcp alignments of tts-data",
    url="http://github.com/malarinv/jasper-asr",
    author="Malar Kannan",
    author_email="malarkannan.invention@gmail.com",
    license="MIT",
    install_requires=requirements,
    extras_require=extra_requirements,
    packages=["."],
    entry_points={
        "console_scripts": [
            "jasper_transcribe = jasper.__main__:main",
            "asr_rpyc_server = jasper.server:main",
        ]
    },
    zip_safe=False,
)
