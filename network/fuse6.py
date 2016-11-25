'''
Network symbol with different fusion module.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
from fuse3 import SkipNet

def get_symbol(num_classes, num_depth, widen_factor):
    net=SkipNet(6,num_classes, num_depth, widen_factor)
    return net.get_symbol()