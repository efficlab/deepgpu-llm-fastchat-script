#!/bin/bash

# Python3 check.
PY3_PATH=$(which python3)
if [ -z "$PY3_PATH" ]; then
  echo "Python3 is required."
  exit 1
fi

# Download model.
MODEL_DIRECTORY=deepgpu/models
if [ ! -d "$MODEL_DIRECTORY" ]; then
  apt install -y git-lfs
  mkdir -p $MODEL_DIRECTORY
  cd $MODEL_DIRECTORY/
  git-lfs clone https://modelscope.cn/qwen/Qwen1.5-4B-Chat.git
fi

# Install fastchat.
pip3 install jinja2==3.1.2 plotly pydantic==1.10.13 gradio==3.50.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install --upgrade setuptools wheel pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install https://aiacc-inference-public-v2.oss-cn-hangzhou.aliyuncs.com/aiacc-inference-llm/fschat_deepgpu-0.2.31%2Bpt2.1-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple --use-pep517

# Setup ENV.
export DEEPGPU_CB=True

# Start fastchat controller.
python3 -m fastchat.serve.controller --host localhost --port 21001 > /dev/null 2>&1 &

# Start worker for LLM.
CUDA_VISIBLE_DEVICES=0,1 python3 -m fastchat.serve.model_worker --num-gpus 2 --model-names qwen-4b-deepgpu --model-path /root/deepgpu/models/Qwen1.5-4B-Chat --worker http://localhost:21002 --controller-address http://localhost:21001 --host localhost --port 21002 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-names qwen-4b-deepgpu --model-path /root/deepgpu/models/Qwen1.5-4B-Chat --worker http://localhost:21002 --controller-address http://localhost:21001 --host localhost --port 21002 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-names qwen-4b-base --model-path /root/deepgpu/models/Qwen1.5-4B-Chat --worker http://localhost:21003 --controller-address http://localhost:21001 --host localhost --port 21003 > /dev/null 2>&1 &

# Start Web service based on gradio.
# python3 -m fastchat.serve.gradio_web_server_multi --controller-url http://localhost:21001 --host 0.0.0.0 --port 5001 --model-list-mode reload > /dev/null 2>&1 &
python3 -m fastchat.serve.gradio_web_server --controller-url http://localhost:21001 --host 0.0.0.0 --port 5001 --model-list-mode reload > /dev/null 2>&1 &

# Show GPU performance monitoring.
watch -n0.1 nvidia-smi
