from setuptools import setup, find_packages
import os
import subprocess

def generate_proto():
    proto_dir = os.path.join(os.path.dirname(__file__), 'proto')
    output_dir = os.path.join(os.path.dirname(__file__), 'voice_assist', 'proto')
    os.makedirs(output_dir, exist_ok=True)
    
    proto_file = os.path.join(proto_dir, 'voice_assist.proto')
    subprocess.run([
        'python', '-m', 'grpc_tools.protoc',
        f'--proto_path={proto_dir}',
        f'--python_out={output_dir}',
        f'--grpc_python_out={output_dir}',
        proto_file
    ], check=True)

# Generate proto files
generate_proto()

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