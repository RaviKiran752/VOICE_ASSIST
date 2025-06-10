from setuptools import setup, find_packages

setup(
    name="voice_assist",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "proto": ["*.py"],
    },
    install_requires=[
        "grpcio",
        "grpcio-tools",
        "protobuf",
    ],
) 