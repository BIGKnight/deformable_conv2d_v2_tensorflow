TF_INC=/home/zzn/SANet_implementation-master/lib/python3.5/site-packages/tensorflow/include
CUDA_HOME=/usr/local/cuda-9.0
nvcc -std=c++11 -c -o deformable_conv2d.cu.o deformable_conv2d.cu.cc -I $TF_INC -I $CUDA_HOME -I /usr/local -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-9.0/lib64 --expt-relaxed-constexpr