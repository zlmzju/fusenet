import mxnet as mx


def get_conv(name, data, kout, kernel, stride, pad, with_relu=True):
    data = mx.symbol.Convolution(name=name+'_conv', data=data, num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    data = mx.symbol.BatchNorm(name=name + '_bn', data=data, fix_gamma=False, momentum=0.9, eps=2e-5)
    if with_relu:
        data=mx.symbol.Activation(name=name + '_relu', data=data, act_type='relu')
    return data

def get_deep(name, data, kin, kout, with_relu=False):
    data = get_conv(name=name+'_conv1', data=data, kout=kout/4, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    data = get_conv(name=name+'_conv2', data=data, kout=kout/4, kernel=(3, 3), stride=(1, 1) if kin==kout or kout==4*kin else (2, 2), pad=(1, 1))
    data = get_conv(name=name+'_conv3', data=data, kout=kout  , kernel=(1, 1), stride=(1, 1), pad=(0, 0), with_relu=with_relu)
    return data

def get_line(name, data, kin, kout):
    if kin!=kout:
        data = get_conv(name=name+'_line', data=data, kout=kout, kernel=(1, 1), stride=(1, 1) if kout==4*kin else (2, 2), pad=(0, 0), with_relu=False)
    return data
def get_fusion(name, data1, data2, kin, kout, last):
    line1= get_line(name+'l1', data1, kin, kout)
    line2= get_line(name+'l2', data2, kin, kout)
    deep1= get_deep(name+'d1', data1, kin, kout)
    deep2= get_deep(name+'d2', data2, kin, kout)

    fuse = 0.5*(line1+line2)
    data1 = fuse+deep1
    data2 = fuse+deep2
    if last:
        data1=fuse+deep1+deep2
        data2=data1
    data1 = mx.symbol.Activation(name=name+'_relu1', data=data1, act_type='relu')
    data2 = mx.symbol.Activation(name=name+'_relu2', data=data2, act_type='relu')
    return data1,data2

def get_group(name,data1,data2,num_block,kin,kout,last=False):
    for idx in range(num_block):
        data1,data2 = get_fusion(name+'_b%d'%(idx+1), data1, data2, kin, kout, last if idx==num_block-1 else False)
        kin=kout
    return data1,data2


def get_symbol(num_classes=1000, net_depth=101):
    # setup model parameters
    model_cfgs = {
        50 : (2, 2, 4, 1), #actual depth is 29, corresponds to ResNet-50:  (3, 4,  6, 3)
        101: (2, 2, 8, 2), #actual depth is 44, corresponds to ResNet-101: (3, 4, 23, 3)
        152: (2, 4,15, 2), #actual depth is 71, corresponds to ResNet-101: (3, 8, 36, 3)
    }
    blocks_num = model_cfgs[net_depth]
    
    # start network definition
    data = mx.symbol.Variable(name='data')
    # stage conv1_x
    data = get_conv(name='g0_conv', data=data, kout=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3))
    data = mx.symbol.Pooling(name='g0_pool', data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    # stage conv2_x, conv3_x, conv4_x, conv5_x
    data1, data2=get_group('g1', data , data , num_block=blocks_num[0], kin=64,  kout=64*4)
    data1, data2=get_group('g2', data1, data2, num_block=blocks_num[1], kin=256, kout=128*4)
    data1, data2=get_group('g3', data1, data2, num_block=blocks_num[2], kin=512, kout=256*4)
    data ,   _  =get_group('g4', data1, data2, num_block=blocks_num[3], kin=1024, kout=512*4,last=True)
    avg = mx.symbol.Pooling(name='global_pool', data=data, kernel=(7, 7), stride=(1, 1), pool_type='avg')
    flatten = mx.sym.Flatten(name="flatten", data=avg)
    fc = mx.symbol.FullyConnected(name='fc_score', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=fc)
    
    return softmax