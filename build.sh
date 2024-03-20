#!/bin/bash

# Install Pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
echo 'export PATH=$PATH:/home/truffle/.local/bin' >> ~/.bashrc
source ~/.bashrc

# Install dependencies
LLVM_VERSION="17" # Replace with desired version
wget https://apt.llvm.org/llvm.sh && \
  sudo chmod +x llvm.sh && \
  sudo ./llvm.sh ${LLVM_VERSION} all && \
  sudo ln -s /usr/bin/llvm-config-* /usr/bin/llvm-config
sudo apt-get update && \
  sudo apt-get install -y --no-install-recommends \
    libzstd-dev && \
  sudo rm -rf /var/lib/apt/lists/* && \
  sudo apt-get clean

curl https://sh.rustup.rs -sSf | sh && source "$HOME/.cargo/env"

# Clone source code
git clone https://github.com/mlc-ai/mlc-llm
cd mlc-llm

# Checkout specific version and update submodules
VERSION="latest_sha" # Replace with desired version (from config.py)
git checkout ${VERSION}
git submodule update --init --recursive

# Apply patch
PATCH_FILE="patches/patch.diff" # Replace with appropriate patch file
cp ${PATCH_FILE} .
git apply ${PATCH_FILE}

# Build MLC LLM
mkdir build && cd build
cmake -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 -DCMAKE_CUDA_ARCHITECTURES="87" -DUSE_CUDA=ON -DUSE_TENSORRT_CODEGEN=ON -DUSE_CUDNN=ON -DUSE_CUBLAS=ON -DUSE_CURAND=ON -DUSE_CUTLASS=ON -DUSE_THRUST=ON -DUSE_GRAPH_EXECUTOR_CUDA_GRAPH=ON -DUSE_STACKVM_RUNTIME=ON -DUSE_LLVM="/usr/bin/llvm-config --link-static" -DHIDE_PRIVATE_SYMBOLS=ON -DSUMMARIZE=ON ../ && ninja

rm -rf CMakeFiles tvm/CMakeFiles tokenizers/CMakeFiles tokenizers/release

# Build TVM python module
export TVM_LIBRARY_PATH=$(pwd)/tvm
cd ../3rdparty/tvm/python
python3 setup.py --verbose bdist_wheel
cp dist/tvm*.whl ../..
rm -rf dist build

# Install TVM wheel
cd ../..
pip3 install --no-cache-dir --force-reinstall --verbose tvm*.whl

# Build mlc-llm python module
cd ..
python3 setup.py --verbose bdist_wheel
cp dist/mlc*.whl ./

cd python
python3 setup.py --verbose bdist_wheel
cp dist/mlc*.whl ../

# Install mlc-llm wheels
cd ../
pip3 install --no-cache-dir --force-reinstall --verbose mlc*.whl

# Install transformers version
pip3 install --no-cache-dir --verbose 'transformers<4.36'

# Verify installation
pip3 show mlc_llm
python3 -m mlc_llm.build --help
python3 -c "from mlc_chat import ChatModule; print(ChatModule)"

# Additional steps (optional)
# - Set environment variables (e.g., TVM_HOME)
# - Copy benchmark script
ln -s '$HOME/mlc-llm/3rdparty/tvm/3rdparty' $(pip3 show torch | grep Location: | cut -d' ' -f2)/tvm/3rdparty
# Append TVM_HOME to ~/.bashrc
echo 'export TVM_HOME=$HOME/mlc-llm/3rdparty/tvm' >> ~/.bashrc

# Source ~/.bashrc to make the variable available in the current session
source ~/.bashrc
echo "MLC LLM built successfully!"