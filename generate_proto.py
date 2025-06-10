import os
import subprocess
import sys

def generate_proto():
    proto_dir = os.path.join(os.path.dirname(__file__), 'proto')
    output_dir = os.path.join(os.path.dirname(__file__), 'voice_assist', 'proto')
    os.makedirs(output_dir, exist_ok=True)
    
    proto_file = os.path.join(proto_dir, 'voice_assist.proto')
    subprocess.check_call([
        sys.executable, '-m', 'grpc_tools.protoc',
        f'--proto_path={proto_dir}',
        f'--python_out={output_dir}',
        f'--grpc_python_out={output_dir}',
        proto_file
    ])

if __name__ == '__main__':
    generate_proto() 