'''
Network symbol with different fusion module.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
from base import FuseNet

class ResNet(FuseNet):
    def __init__(self, *args, **kwargs):
        FuseNet.__init__(self, *args, **kwargs)
        self.set_blocks(block_depth=2)
    #different network has different fusion module
    def get_fusion(self, name, data, kin, kout):
        line= self.get_zero(name+'_p0', data, kin, kout)
        two= self.get_two(name+'_p2', data, kin, kout)
        #resnet style: identity + two convs
        data = line+two
        data = mx.symbol.Activation(name=name+'_relu', data=data, act_type='relu')
        return data

def get_symbol(num_classes=10, num_depth=50, widen_factor=1):
    net=ResNet(num_classes,num_depth, widen_factor)
    return net.get_symbol()