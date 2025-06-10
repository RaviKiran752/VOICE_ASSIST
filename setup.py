from setuptools import setup, find_packages

setup(
    name="voice_assist",
    version="0.1.0",
    packages=find_packages(include=['voice_assist', 'voice_assist.*']),
    package_data={
        "voice_assist.proto": ["*.py"],
    },
    install_requires=[
        "grpcio==1.60.0",
        "grpcio-tools==1.60.0",
        "protobuf==4.25.1",
        "fastapi==0.109.0",
        "uvicorn==0.27.0",
        "TTS==0.21.1",
        "numpy==1.22.0",
        "torch==2.1.2",
        "python-multipart==0.0.6",
        "soundfile==0.12.1",
        "redis==5.0.1",
        "prometheus-client==0.19.0",
    ],
) 