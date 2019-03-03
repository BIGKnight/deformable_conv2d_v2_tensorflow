TF_INC=/home/zzn/SANet_implementation-master/lib/python3.5/site-packages/tensorflow/include
TF_CFLAGS=-I/home/zzn/SANet_implementation-master/lib/python3.5/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0
TF_LFLAGS=-L/home/zzn/SANet_implementation-master/lib/python3.5/site-packages/tensorflow -ltensorflow_framework
CUDA_HOME=/usr/local/cuda-9.0
if [ ! -f $TF_INC/tensorflow/stream_executor/cuda/cuda_config.h ]; then
    cp ./cuda_config.h $TF_INC/tensorflow/stream_executor/cuda/
fi
g++ -std=c++11 -shared -o deformable_conv2d.so deformable_conv2d.cc deformable_conv2d.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors -I $CUDA_HOME/include -D_GLIBCXX_USE_CXX11_ABI=0