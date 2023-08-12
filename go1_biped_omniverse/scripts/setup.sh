## Torch
wget https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl -O torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
rm torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip3 install torchvision==0.15.1
sudo apt install -y libopenblas-base libopenmpi-dev libomp-dev

## ONNX
wget https://nvidia.box.com/shared/static/v59xkrnvederwewo2f1jtv6yurl92xso.whl -O onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
rm onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl
pip3 install onnx==1.12.0
pip3 install protobuf==3.20.3
pip3 install torchvision==v0.15.1

