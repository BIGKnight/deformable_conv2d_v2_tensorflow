#!/usr/bin/env bash
TF_CFLAGS=(-I/home/zzn/SANet_implementation-master/lib/python3.5/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0)
TF_LFLAGS=(-L/home/zzn/SANet_implementation-master/lib/python3.5/site-packages/tensorflow -ltensorflow_framework)
CUDA_HOME=/usr/local/cuda-9.0
nvcc -std=c++11 -c -o deformable_conv2d.cu.o deformable_conv2d.cu.cc -I $CUDA_HOME -I /usr/local ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-9.0/lib64 --expt-relaxed-constexpr
g++ -std=c++11 -shared -o deformable_conv2d.so deformable_conv2d.cc deformable_conv2d.cu.o -L /usr/local/cuda-9.0/lib64 -I $CUDA_HOME/include -D GOOGLE_CUDA=1  ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
