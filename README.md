# Deformable_Conv2d_Tensorflow
deformable_conv2d layer implemented in Tensorflow.
<br>besides, I also complete an example net, [here](https://github.com/BIGKnight/ADCrowdNet_tensorflow_implementation)
and I'm very sorry that I did not implement the swapaxis methods, so the im2col_step parameter are only allowed using value one.
## ENVIRONMENT CONFIGURATION
1. OS: ubuntu16.04 <br>
2. GPU: 1 gtx1080Ti <br>
3. LANGUAGE: python3.5 & c++11 & cuda c<br>
4. DL FRAMEWORK: tensorflow_gpu 1.12.0<br>
5. ANCILLARY LIB: numpy: 1.15.4, etc
6. GPU API: NVIDIA CUDA 9.0 & cuDNN 7.0
7. COMPILE: nvcc & gcc 5.4.0
## INSTALL PROCEDURE
1. cd "current project"
2. run make.sh<br>tips: you need to modify the path parameters first. and all the -I and -L path in the nvcc and g++ orders need to be checked, make sure they are the correct path in your system<br>
3. load the dynamic lib which created by the make.sh, and do the python wrapper(see the script:deformable_conv2d.py)
