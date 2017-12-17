git config --global alias.ss 'status -s'
pip install --user tensorflow-gpu
pip install --user pandas
pip install --user sklearn
pip install --user matplotlib

#setup
mkdir -p tools-bin/cuda
mkdir -p tools-bin/cudnn
cat tools/cudnn/cudnn-part* > tools-bin/cudnn/cudnn-8.0-linux-x64-v6.0.tgz
tar xvf tools-bin/cudnn/cudnn-8.0-linux-x64-v6.0.tgz -C tools-bin/cudnn
CUR_DIR=$(pwd)
CUDA_DIR="$CUR_DIR/tools-bin/cuda"
echo "install cuda to this dir $CUDA_DIR\n"
python download-cuda.py
bash ./tools-bin/cuda_9.1.85_387.26_linux.run
mv tools-bin/cudnn/cuda/include/cudnn.h tools-bin/cuda/include/
mv tools-bin/cudnn/cuda/lib64/* tools-bin/cuda/lib64/
# env
echo "add to profile export PATH=\"$CUDA_DIR/bin:\$PATH\""
echo "export PATH=\"$CUDA_DIR/bin:\$PATH\"" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\"$CUDA_DIR/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
. ~/.bashrc
