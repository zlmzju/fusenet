'''
Network symbol with different fusion module.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
from fuse3 import SkipNet

def get_symbol(num_classes=10, num_depth=50, widen_factor=1):
    net=SkipNet(6,num_classes, num_depth, widen_factor)
    return net.get_symbol()