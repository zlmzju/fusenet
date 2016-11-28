import mxnet as mx

def get_conv(name, data, kout, kernel, stride, pad, with_relu=True):
    conv = mx.symbol.Convolution(name=name, data=data, num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    bn = mx.symbol.BatchNorm(name=name + '_bn', data=conv, fix_gamma=False, momentum=0.9, eps=2e-5)
    return (mx.symbol.Activation(name=name + '_relu', data=bn, act_type='relu') if with_relu else bn)

def get_deep(name, data, kin, kout, stride,with_relu=False):
    conv1 = get_conv(name=name+'_conv1', data=data , kout=kout/4, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    conv2 = get_conv(name=name+'_conv2', data=conv1, kout=kout/4, kernel=(3, 3), stride=stride, pad=(1, 1))
    conv3 = get_conv(name=name+'_conv3', data=conv2, kout=kout  , kernel=(1, 1), stride=(1, 1), pad=(0, 0), with_relu=with_relu)
    return conv3
    
def get_shallow(name, data, kin, kout, stride):
    if kin==kout:
        shallow = data
    else:
        shallow = get_conv(name=name+'_proj', data=data, kout=kout, kernel=(1, 1), stride=stride, pad=(0, 0), with_relu=False)

    return shallow
    
def get_fusion(name, data, kin, kout, stride):
    shallow= get_shallow(name, data, kin, kout, stride)
    deep   = get_deep(name, data, kin, kout, stride)

    fusion = shallow + deep
    fusion = mx.symbol.Activation(name=name+'_relu', data=fusion, act_type='relu')

    return fusion

def get_group(name,data,num_block,kin,kout,stride):
    for idx in range(num_block):
        data = get_fusion(name=name+'_b%d'%(idx+1), data=data, kin=kin, kout=kout, 
                                stride= stride if idx == 0 else (1, 1))
        kin=kout
    return data


def get_symbol(num_classes=1000, net_depth=101):
    # setup model parameters
    model_cfgs = {
        50:  (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }
    blocks_num = model_cfgs[net_depth]
    
    # start network definition
    data = mx.symbol.Variable(name='data')
    # stage conv1_x
    conv1 = get_conv(name='g0', data=data, kout=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3))
    pool1 = mx.symbol.Pooling(name='g0_pool', data=conv1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    # stage conv2_x, conv3_x, conv4_x, conv5_x
    conv2_x=get_group(name='g1', data=pool1  , num_block=blocks_num[0], kin=64,  kout=64*4, stride=(1,1))
    conv3_x=get_group(name='g2', data=conv2_x, num_block=blocks_num[1], kin=256, kout=128*4, stride=(2,2))
    conv4_x=get_group(name='g3', data=conv3_x, num_block=blocks_num[2], kin=512, kout=256*4, stride=(2,2))
    conv5_x=get_group(name='g4', data=conv4_x, num_block=blocks_num[3], kin=1024, kout=512*4, stride=(2,2))
    
    avg = mx.symbol.Pooling(name='global_pool', data=conv5_x, kernel=(7, 7), stride=(1, 1), pool_type='avg')
    flatten = mx.sym.Flatten(name="flatten", data=avg)
    fc = mx.symbol.FullyConnected(name='fc_score', data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(name='softmax', data=fc)
    
    return softmax