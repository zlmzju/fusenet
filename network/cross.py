'''
Network symbol with different fusion module.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
from base import FuseNet

class CrossNet(FuseNet):
    def __init__(self, *args, **kwargs):
        FuseNet.__init__(self, *args, **kwargs)
        self.set_blocks(block_depth=4)
        
    #different network has different fusion module
    def get_fusion(self, name, data1, data2, kin, kout, last_relu=True):
        line1= self.get_zero(name+'_l1', data1, kin, kout)
        line2= self.get_zero(name+'_l2', data2, kin, kout)
        deep1= self.get_two(name+'_d1', data1, kin, kout)
        deep2= self.get_two(name+'_d2', data2, kin, kout)

        fuse = line1+line2
        fuse = 0.5*fuse
        data1 = 1.0*(fuse+deep1)
        data2 = 1.0*(fuse+deep2)
        if last_relu:
            data1 = mx.sym.Activation(name=name+'_relu1', data=data1, act_type='relu')
            data2 = mx.sym.Activation(name=name+'_relu2', data=data2, act_type='relu')
        return data1,data2

    def get_group(self, name,data1,data2,count,kin,kout, last_relu=True):
        for idx in range(count):
            data1,data2 = self.get_fusion(name+'_b%d'%(idx+1), data1, data2, kin, kout, last_relu if idx==count-1 else True)
            kin=kout
        return data1,data2

    def get_symbol(self, num_classes=10, net_depth=110,widen_factor=4):
        # start network definition
        data = mx.sym.Variable(name='data')
        # stage conv1_x
        data=self.get_conv('g0', data, self.num_filters[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        # stage conv2_x, conv3_x, conv4_x, conv5_x
        data1,data2=self.get_group('g1', data , data , self.num_blocks[0], self.num_filters[0], self.num_filters[1])
        data1,data2=self.get_group('g2', data1, data2, self.num_blocks[1], self.num_filters[1], self.num_filters[2])
        data1,data2=self.get_group('g3', data1, data2, self.num_blocks[2], self.num_filters[2], self.num_filters[3], last_relu=False)
        fusion=data1+data2
        data=mx.sym.Activation(name='last_relu', data=fusion, act_type='relu')
        # classification layer
        data=self.get_fc('cls', data)
        softmax = mx.sym.SoftmaxOutput(name='softmax', data=data)
        return softmax

def get_symbol(num_classes=10, num_depth=50, widen_factor=1):
    net=CrossNet(num_classes,num_depth, widen_factor)
    return net.get_symbol()