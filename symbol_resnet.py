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

def get_shortcut(name, data, kin, kout, stride):
    shortcut = data
    return shortcut
    
def get_fusion(name, data, kin, kout, stride):
    if kin == kout:
        shortcut1= get_shortcut(name+'_shortcut1', data, kin, kout, stride)
        deep1 = get_deep2bott(name+'_deep1', data, kin, kout, stride)
        data = deep1 + shortcut1
        data = mx.symbol.Activation(name=name+'_relu', data=data, act_type='relu')
    else:
        data = get_deep2bott(name+'_deep1', data, kin, kout, stride)
        data = mx.symbol.Activation(name=name+'_relu', data=data, act_type='relu')

    return data

def get_group(name,data,num_block,kin,kout,stride):
    for idx in range(num_block):
        data = get_fusion(name=name+'_b%d'%(idx+1), data=data, kin=kin, kout=kout, stride= stride if idx == 0 else (1, 1))
        kin=kout
    return data

def get_symbol(num_classes=1000):
    # setup model parameters
    blocks_num = (2,4,22,4)
    channels = 64
    # start network definition
    data = mx.symbol.Variable(name='data')
    # stage conv1_x
    conv1 = mx.symbol.Convolution(name='g0', data=data, num_filter=channels, kernel=(7,7), stride=(2,2), pad=(3,3), no_bias=True)
    pool1 = mx.symbol.Pooling(name='g0_pool', data=conv1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    # stage conv2_x, conv3_x, conv4_x, conv5_x
    conv2_x=get_group(name='g1', data=pool1 , num_block=blocks_num[0], kin=channels,  kout=channels*4, stride=(1,1))

    conv3_x=get_group(name='g2', data=conv2_x, num_block=blocks_num[1], kin=channels*4, kout=channels*8, stride=(2,2))

    conv4_x=get_group(name='g3', data=conv3_x, num_block=blocks_num[2], kin=channels*8, kout=channels*16, stride=(2,2))

    conv5_x=get_group(name='g4', data=conv4_x, num_block=blocks_num[3], kin=channels*16, kout=channels*32, stride=(2,2))
    
    avg = mx.symbol.Pooling(name='global_pool', data=conv5_x, kernel=(7, 7), stride=(1, 1), pool_type='avg')
    flatten = mx.sym.Flatten(name="flatten", data=avg)
    fc = mx.symbol.FullyConnected(name='fc_score', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=fc)
    
    return softmax  