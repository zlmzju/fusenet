'''
Network symbol with different fusion module.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
from base import FuseNet

class SkipNet(FuseNet):
    def __init__(self, skip=3, *args, **kwargs):
        FuseNet.__init__(self, *args, **kwargs)
        self.block_depth=(self.num_depth-2)/skip
        self.set_blocks(block_depth=self.block_depth)
    
    def get_line(self, name, data, kin, kout, relu=False):
        if kin!=kout or 'g1_b1' in name:    #first line in group1_block1 is conv
            data = self.get_one(name+'_line', data, kin, kout, relu)
        return data

    #different network has different fusion module
    def get_fusion(self, name, data, kin, kout):
        left= self.get_line(name+'_p0', data, kin, kout)
        for i in range(self.block_depth):
            data= self.get_one(name+'_p2_%d'%i, data, kin, kout, relu=False if i==self.block_depth-1 else True)
            kin=kout
        #resnet style: identity + two convs
        data = left+data
        data = mx.symbol.Activation(name=name+'_relu', data=data, act_type='relu')
        return data

def get_symbol(num_classes=10, num_depth=50, widen_factor=1):
    net=SkipNet(3,num_classes, num_depth, widen_factor)
    return net.get_symbol()