import mxnet as mx
import math
import random


def get_conv(name, data, kout, kernel, stride, pad, relu=True):
    conv = mx.symbol.Convolution(name=name, data=data, num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    bn = mx.symbol.BatchNorm(name=name + '_bn', data=conv, fix_gamma=False, momentum=0.9, eps=2e-5)
    return (mx.symbol.Activation(name=name + '_relu', data=bn, act_type='relu') if relu else bn)

def get_pre(name, data, kout, kernel, stride, pad, relu=True):
    data = mx.symbol.BatchNorm(name=name + '_bn', data=data, fix_gamma=False, momentum=0.9, eps=2e-5)
    data = mx.symbol.Activation(name=name + '_relu', data=data, act_type='relu')
    conv = mx.symbol.Convolution(name=name, data=data, num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    return conv

def get_deep(name, data, kin, kout, stride,relu=True):
    conv1 = get_conv(name=name+'_conv1', data=data , kout=kout, kernel=(3, 3), stride=stride, pad=(1, 1))
    conv2 = get_conv(name=name+'_conv2', data=conv1, kout=kout, kernel=(3, 3), stride=(1, 1), pad=(1, 1),relu=relu)
    return conv2

def get_deep2(name, data, kin, kout, stride,relu=True):
    conv = mx.symbol.Convolution(name=name+'_conv1', data=data, num_filter=kout, kernel=(3,3), stride=stride, pad=(1,1), no_bias=True)
    conv = mx.symbol.BatchNorm(name=name + '_bn1', data=conv, fix_gamma=False, momentum=0.9, eps=2e-5)
    conv = mx.symbol.Activation(name=name + '_relu1', data=conv, act_type='relu')
    
    conv = mx.symbol.Convolution(name=name+'_conv2', data=conv, num_filter=kout, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=True)
    conv = mx.symbol.BatchNorm(name=name + '_bn2', data=conv, fix_gamma=False, momentum=0.9, eps=2e-5)
    return conv

def get_deep2bott(name, data, kin, kout, stride,relu=True):
    conv = mx.symbol.Convolution(name=name+'_conv1', data=data, num_filter=int(kout/4), kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True)
    conv = mx.symbol.BatchNorm(name=name + '_bn1', data=conv, fix_gamma=False, momentum=0.9, eps=2e-5)
    conv = mx.symbol.Activation(name=name + '_relu1', data=conv, act_type='relu')
    conv = mx.symbol.Convolution(name=name+'_conv2', data=conv, num_filter=int(kout/4), kernel=(3,3), stride=stride, pad=(1,1), no_bias=True)
    conv = mx.symbol.BatchNorm(name=name + '_bn2', data=conv, fix_gamma=False, momentum=0.9, eps=2e-5)
    conv = mx.symbol.Activation(name=name + '_relu2', data=conv, act_type='relu')
    conv = mx.symbol.Convolution(name=name+'_conv3', data=conv, num_filter=kout, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True)
    conv = mx.symbol.BatchNorm(name=name + '_bn3', data=conv, fix_gamma=False, momentum=0.9, eps=2e-5)
    
    return conv
    
    
def get_fusion(name, datal, datar, kin, kout, stride):
    if kin == kout:
        shortcut1l = datal
        deep1l = get_deep2bott(name+'_deep1l', datal, kin, kout, stride)

        shortcut1r = datar
        deep1r = get_deep2bott(name+'_deep1r', datar, kin, kout, stride)

        fuse = 0.5 * (shortcut1l + shortcut1r)

        datal=deep1l+fuse
        datar=deep1r+fuse
    else:
        datal = get_deep2bott(name+'_deep1l', datal, kin, kout, stride)
        datar = get_deep2bott(name+'_deep1r', datar, kin, kout, stride)


    datal = mx.symbol.Activation(name=name+'_relu', data=datal, act_type='relu')
    datar = mx.symbol.Activation(name=name+'_relu', data=datar, act_type='relu')
    return datal, datar

def get_group(name, datal, datar, num_block,  kin, kout, stride):
    for idx in range(num_block):
        datal, datar = get_fusion(name=name+'_b%d'%(idx+1), datal=datal, datar=datar, kin=kin, kout=kout, stride=stride if idx == 0 else (1, 1))
        kin = kout
    return datal, datar

def get_symbol(num_classes=1000):
    # setup model parameters
    blocks_num = (1,2,11,2)
    channels = 64
    # start network definition
    data = mx.symbol.Variable(name='data')
    # stage conv1_x
    conv1 = mx.symbol.Convolution(name='g0', data=data, num_filter=channels, kernel=(7,7), stride=(2,2), pad=(3,3), no_bias=True)
    pool1 = mx.symbol.Pooling(name='g0_pool', data=conv1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    # stage conv2_x, conv3_x, conv4_x, conv5_x
    conv2_x_1, conv2_x_2 = get_group(name='g1', datal=pool1, datar=pool1, num_block=blocks_num[0], kin=channels,  kout=channels*4, stride=(1,1))

    conv3_x_1, conv3_x_2 = get_group(name='g2', datal=conv2_x_1, datar=conv2_x_2, num_block=blocks_num[1], kin=channels*4, kout=channels*8, stride=(2,2))

    conv4_x_1, conv4_x_2 = get_group(name='g3', datal=conv3_x_1, datar=conv3_x_2, num_block=blocks_num[2], kin=channels*8, kout=channels*16, stride=(2,2))

    conv5_x_1, conv5_x_2 = get_group(name='g4', datal=conv4_x_1, datar=conv4_x_2, num_block=blocks_num[3], kin=channels*16, kout=channels*32, stride=(2,2))
    
    conv5_x= mx.symbol.Concat(conv5_x_1,conv5_x_2)
    avg = mx.symbol.Pooling(name='global_pool', data=conv5_x, kernel=(7, 7), stride=(1, 1), pool_type='avg')
    flatten = mx.sym.Flatten(name="flatten", data=avg)
    fc = mx.symbol.FullyConnected(name='fc_score', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=fc)
    
    return softmax