from setuptools import setup, find_packages

setup(
    name="voice_assist",
    version="0.1.0",
    packages=find_packages(include=['voice_assist', 'voice_assist.*']),
    package_data={
        "voice_assist.proto": ["*.py"],
    },
    install_requires=[
        "grpcio",
        "grpcio-tools",
        "protobuf",
        "fastapi",
        "uvicorn",
        "TTS",
        "numpy",
        "torch",
        "python-multipart",
        "soundfile",
        "redis",
        "prometheus-client",
    ],
) 