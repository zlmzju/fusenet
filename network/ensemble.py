'''
Network symbol for ensemble.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx

def get_conv(name, data, kout, kernel, stride, pad, weights, with_relu=True):
    conv = mx.symbol.Convolution(name=name, data=data, weight=weights[0], num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True)
    bn = mx.symbol.BatchNorm(name=name + '_bn', data=conv, fix_gamma=False, momentum=0.99, eps=2e-5, gamma=weights[1], beta=weights[2])
    return (mx.symbol.Activation(name=name + '_relu', data=bn, act_type='relu') if with_relu else bn)


def get_deep(name, data, kin, kout, weights, with_relu=False):
    data = get_conv(name=name+'_conv3', data=data, kout=kout, kernel=(3, 3), stride=(1,1) if kin==kout else (2, 2), pad=(1, 1), weights=weights[1], with_relu=with_relu)
    return data
    
def get_line(name, data, kin, kout,weights):
    data = get_conv(name=name+'_line', data=data, kout=kout, kernel=(3, 3), stride=(1,1) if kin==kout else (2, 2), pad=(1, 1),weights=weights, with_relu=False)
    return data

def get_group(name,data,num_block,kin,kout,select_left,weights):
    if select_left:
        data=get_line(name+'_skip', data, kin, kout, weights[0])
    else:
        for idx in range(num_block):
            data = get_deep(name=name+'_b%d'%(idx+1), data=data, kin=kin, kout=kout, 
                                    weights=weights[1][idx], with_relu=True if idx<num_block-1 else False)
            kin=kout
    data   = mx.symbol.Activation(name=name+'_relu', data=data, act_type='relu')
    return data

def get_conv_weights(name):
    weight=mx.symbol.Variable(name= name + '_weight')
    bn_gamma=mx.symbol.Variable(name= name + '_bn_gamma')
    bn_beta=mx.symbol.Variable(name= name + '_bn_beta')
    return [weight,bn_gamma,bn_beta]

def get_group_weights(name,num_block):
    #group 1
    left=get_conv_weights(name+'_left')
    rights=[]
    for idx in range(num_block):
        right1=get_conv_weights(name+'_b%d'%(idx+1)+'_right1')
        right2=get_conv_weights(name+'_b%d'%(idx+1)+'_right2')
        rights.append((right1,right2))
    return (left,rights)


def get_softmax(name,data,num_classes,fc_weights):
    avg = mx.symbol.Pooling(name=name+'global_pool', data=data, kernel=(8,8), stride=(1, 1), pool_type='avg')
    flatten = mx.sym.Flatten(name=name+"flatten", data=avg)
    fc = mx.symbol.FullyConnected(name=name+'fc_score', data=flatten, num_hidden=num_classes,weight=fc_weights[0],bias=fc_weights[1])
    #softmax = mx.symbol.SoftmaxOutput(name='softmax', data=fc)
    return fc#, softmax


def get_symbol(num_classes=10, net_depth=20,widen_factor=1):
    # setup model parameters
    block1_num=(net_depth-2)/3
    block2_num=(net_depth-2-block1_num)/2
    block3_num=(net_depth-2-(block1_num+block2_num))
    blocks_num=(block1_num,block2_num,block3_num)
    print blocks_num
    if net_depth!=((block1_num+block2_num+block3_num)+2) or block1_num<=0:
        print 'invalid depth number: %d'%net_depth,', blocks numbers: ',blocks_num
        return
    # start network definition
    data = mx.symbol.Variable(name='data')
    # stage conv1_x
    base_weights=get_conv_weights('g0')
    # stage conv2_x, conv3_x, conv4_x, conv5_x

    base_num=0
    fcs=[]
    g1_weights=get_group_weights('g1',blocks_num[0])
    g2_weights=get_group_weights('g2',blocks_num[1])
    g3_weights=get_group_weights('g3',blocks_num[2])
    fc_weights=[mx.symbol.Variable(name= 'fc_weight'),mx.symbol.Variable(name= 'fc_bias')]
    for group3_choice in [True, False]:
        for group2_choice in [True,False]:
            for group1_choice in [True,False]:               
                base_num+=1
                conv1 = get_conv(name='uncouple%02dg0_conv'%base_num, data=data, kout=16, kernel=(3, 3), stride=(1, 1), pad=(1, 1), weights=base_weights)
                conv2_x=get_group(name='uncouple%02dg1'%base_num, data=conv1  , num_block=blocks_num[0], 
                    kin=16*widen_factor, kout=16*widen_factor, select_left=group1_choice,weights=g1_weights)
                conv3_x=get_group(name='uncouple%02dg2'%base_num, data=conv2_x  , num_block=blocks_num[1], 
                    kin=16*widen_factor, kout=32*widen_factor, select_left=group2_choice,weights=g2_weights)
                conv4_x=get_group(name='uncouple%02dg3'%base_num, data=conv3_x  , num_block=blocks_num[2], 
                    kin=32*widen_factor, kout=64*widen_factor, select_left=group3_choice,weights=g3_weights)
                fc=get_softmax('n%d'%base_num,conv4_x,num_classes,fc_weights)
                fcs.append(fc)
    softmaxs=[]
    for idx in range(len(fcs)):
        softmax = mx.symbol.SoftmaxOutput(name='softmax%d'%(idx+1), data=fcs[idx])
        softmaxs.append(softmax)

    out=mx.symbol.Group(softmaxs)
    print 'networks number:',base_num
    
    return out