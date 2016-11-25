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
    def get_fusion(self, name, data1, data2, kin, kout, last=False):
        line1= self.get_zero(name+'_l1', data1, kin, kout)
        line2= self.get_zero(name+'_l2', data2, kin, kout)
        deep1= self.get_two(name+'_d1', data1, kin, kout)
        deep2= self.get_two(name+'_d2', data2, kin, kout)

        fuse = 0.5*(line1+line2)
        data1 = fuse+deep1
        data2 = fuse+deep2

        if last: #the last block before fc, data2 is ignored
            data1 = fuse+deep1+deep2
            data2 = data1

        data1 = mx.sym.Activation(name=name+'_relu1', data=data1, act_type='relu')
        data2 = mx.sym.Activation(name=name+'_relu2', data=data2, act_type='relu')
        return data1,data2

    def get_group(self, name,data1,data2,count,kin,kout, last=False):
        for idx in range(count):
            data1,data2 = self.get_fusion(name+'_b%d'%(idx+1), data1, data2, kin, kout, last if idx==count-1 else False)
            kin=kout
        return data1,data2

    def get_symbol(self):
        # start network definition
        data = mx.sym.Variable(name='data')
        # stage conv1_x
        data=self.get_conv('g0', data, self.num_filters[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        # stage conv2_x, conv3_x, conv4_x, conv5_x
        data1,data2=self.get_group('g1', data , data , self.num_blocks[0], self.num_filters[0], self.num_filters[1])
        data1,data2=self.get_group('g2', data1, data2, self.num_blocks[1], self.num_filters[1], self.num_filters[2])
        data ,  _  =self.get_group('g3', data1, data2, self.num_blocks[2], self.num_filters[2], self.num_filters[3], last=True)
        # classification layer
        data=self.get_fc('cls', data)
        softmax = mx.sym.SoftmaxOutput(name='softmax', data=data)
        return softmax

def get_symbol(num_classes, num_depth, widen_factor):
    net=CrossNet(num_classes,num_depth, widen_factor)
    return net.get_symbol()