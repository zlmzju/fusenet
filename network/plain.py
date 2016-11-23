'''
Network symbol with different fusion module.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
from base import FuseNet

class PlainNet(FuseNet):
    def __init__(self, *args, **kwargs):
        FuseNet.__init__(self, *args, **kwargs)
        self.set_blocks(block_depth=1)
    #different network has different fusion module
    def get_fusion(self, name, data, kin, kout):
        data= self.get_one(name+'_p1', data, kin, kout, relu=True)
        return data

def get_symbol(num_classes=10, num_depth=50, widen_factor=1):
    net=PlainNet(num_classes,num_depth, widen_factor)
    return net.get_symbol()