import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
# from deformable_conv2d_op.deformable_conv2d import deformable_conv2d_op
import os.path as osp
import math
import numpy as np
from tensorflow.python.framework import ops

model_path = '/home/zzn/PycharmProjects/ADCrowdNet/vgg_pre-trained_variable_list/vgg_16.ckpt'

# filename = osp.join(osp.dirname(__file__), 'deformable_conv2d.so')
deformable_conv2d_module = tf.load_op_library('./deformable_conv2d.so')
deformable_conv2d_op = deformable_conv2d_module.deformable_conv2d
deformable_conv2d_grad_op = deformable_conv2d_module.deformable_conv2d_back_prop


@ops.RegisterGradient("DeformableConv2D")
def _deformable_conv2d_back_prop(op, grad):
    """The gradients for `deform_conv`.
        Args:
        op: The `deform_conv` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `roi_pool` op.
        Returns:
        Gradients with respect to the input of `deform_conv`.
    """
    data = op.inputs[0]
    filter = op.inputs[1]
    offset = op.inputs[2]
    mask = op.inputs[3]
    '''
        .Attr("strides: list(int)")
        .Attr("num_groups: int")
        .Attr("deformable_groups: int")
        .Attr("im2col_step: int")
        .Attr("no_bias: bool = true")
        .Attr(GetPaddingAttrString())
        .Attr("data_format: {'NCHW' } = 'NCHW' ")
        .Attr("dilations: list(int) = [1, 1, 1, 1]")
    '''
    strides = op.get_attr('strides')
    dilations = op.get_attr('dilations')
    data_format = op.get_attr('data_format')
    im2col_step = op.get_attr('im2col_step')
    no_bias = op.get_attr('no_bias')
    pads = op.get_attr('padding')
    num_groups = op.get_attr('num_groups')
    deformable_groups = op.get_attr('deformable_groups')
    '''
    REGISTER_OP("DeformableConv2DBackProp")
        .Input("input: T")
        .Input("filter: T")
        .Input("offset: T")
        .Input("mask: T")
        .Input("out_grad: T")
        .Output("x_grad: T")
        .Output("filter_grad: T")
        .Output("offset_grad: T")
        .Output("mask_grad: T")
        .Attr("T: {float, double}")
        .Attr("strides: list(int)")
        .Attr("num_groups: int")
        .Attr("deformable_groups: int")
        .Attr("im2col_step: int")
        .Attr("no_bias: bool = true")
        .Attr(GetPaddingAttrString())
        .Attr("data_format: { 'NCHW' } = 'NCHW' ")
        .Attr("dilations: list(int) = [1, 1, 1, 1]")
    '''
    # compute gradient
    data_grad = deformable_conv2d_grad_op(
        input=data,
        filter=filter,
        offset=offset,
        mask=mask,

        out_grad=grad,
        strides=strides,
        num_groups=num_groups, deformable_groups=deformable_groups, im2col_step=im2col_step,
        no_bias=no_bias,
        padding=pads,
        data_format=data_format,
        dilations=dilations)
    return data_grad # List of 4 Tensor, since we have 4 input


if __name__ == '__main__':
    input = tf.constant([1. for i in range(75)], shape=[1, 512, 48, 64])
    # filter = tf.constant([1. for i in range(27)], shape=[256, 512, 3, 3])
    min = - math.sqrt(6. / (7. * 7. * 512.))
    max = math.sqrt(6. / (7. * 7. * 512.))
    # filter = tf.get_variable(name='weight_conv', shape=[3, 3, 512, 256], initializer=tf.random_uniform_initializer(min, max))
    filter = tf.get_variable(name='weight_conv',
                             shape=[5, 5, 512, 256], initializer=tf.ones_initializer)
    filter_deform = tf.transpose(filter, [3, 2, 0, 1])

    offset = tf.constant([0. for i in range(50 * 48 * 64)], shape=[1, 25 * 2, 48, 64])
    mask = tf.constant([1. for i in range(25*48*64)], shape=[1, 25, 48, 64])
    result = deformable_conv2d_op(
        input=input,
        filter=filter_deform,
        offset=offset,
        mask=mask,
        strides=[1, 1, 1, 1],
        num_groups=1,
        deformable_groups=1,
        im2col_step=1,
        no_bias=True,
        padding='SAME',
        data_format='NCHW',
        dilations=[1, 1, 1, 1])

    conv2d = tf.nn.conv2d(input, filter, [1, 1, 1, 1], 'SAME', data_format='NCHW')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        a = sess.run(result)
        b = sess.run(conv2d)
        print(a)
        print(np.mean(a))
        print('------------------------------------/n----------------------------------')
        print(b)
        print(np.mean(b))