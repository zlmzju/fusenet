'''
Network symbol with different fusion module.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
from base import FuseNet

class SideNet(FuseNet):
    def __init__(self, *args, **kwargs):
        FuseNet.__init__(self, *args, **kwargs)
        self.set_blocks(block_depth=4)
    #different network has different fusion module
    def get_fusion(self, name, data, kin, kout):
        two1 = self.get_two(name+'_p1', data, kin, kout)
        two2 = self.get_two(name+'_p2', data, kin, kout)
        #resnet style: identity + two convs
        data = two1+two2
        data = mx.symbol.Activation(name=name+'_relu', data=data, act_type='relu')
        return data

def get_symbol(num_classes, num_depth, widen_factor):
    net=SideNet(num_classes,num_depth, widen_factor)
    return net.get_symbol()