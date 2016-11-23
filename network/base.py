'''
Base network class.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx

class FuseNet(object):
    def __init__(self,num_classes=10, num_depth=50, widen_factor=1):
        self.num_classes=num_classes        #classification number (cifar10 is 10, cifar100 is 100)
        self.num_depth=num_depth            #corresponding plain network depth
        self.num_channels=16*widen_factor   #first conv channels number
        self.set_blocks(block_depth=2, num_group=3) #self.num_blocks: each block 2 layers, 3 groups of blocks

    def get_conv(self, name, data, kout, kernel, stride, pad, relu=True):
        #Conv-BN-ReLU style
        data=mx.sym.Convolution(name=name+'_conv', data=data, num_filter=kout, kernel=kernel, stride=stride, pad=pad, no_bias=True)
        data=mx.sym.BatchNorm(name=name + '_bn', data=data, fix_gamma=False, momentum=0.99, eps=2e-5)
        if relu:
            data=mx.sym.Activation(name=name + '_relu', data=data, act_type='relu')
        return data

    def get_one(self, name, data, kin, kout, relu=True):
        data = self.get_conv(name, data,  kout, kernel=(3, 3), stride=(1,1) if kin==kout else (2, 2), pad=(1, 1), relu=relu)
        return data

    def get_two(self, name, data, kin, kout, relu=False):
        data = self.get_one(name+'_two1', data, kin, kout)
        data = self.get_one(name+'_two2', data, kout, kout, relu)
        return data  

    #identity shortcut
    def get_zero(self, name, data, kin, kout, relu=False):
        if kin!=kout:
            data = self.get_conv(name+'_line', data, kout, kernel=(1, 1), stride=(2, 2), pad=(0, 0), relu=False)
        return data

    def get_fusion(self, name, data, kin, kout):
        #different networks have different fusion module
        pass

    def get_group(self, name, data, count, kin, kout):
        for idx in range(count):
            data = self.get_fusion(name=name+'_b%d'%(idx+1), data=data, kin=kin, kout=kout)
            kin=kout
        return data

    def get_fc(self, name, data):
        avg = mx.sym.Pooling(name=name+'_pool', data=data, kernel=(8, 8), stride=(1, 1), pool_type='avg', global_pool=True)
        flatten = mx.sym.Flatten(name=name+"_flatten", data=avg)
        fc = mx.sym.FullyConnected(name=name+'_fc', data=flatten, num_hidden=self.num_classes)
        return fc

    #caculate the number of blocks for each group
    def set_blocks(self, block_depth=2, num_group=3):
        self.num_blocks=[]
        # based on depth of each basic fusion block
        exclude_depth=2
        for i in range(num_group):
            cur_num=(self.num_depth-exclude_depth)/(block_depth*(num_group-i))
            exclude_depth+=cur_num*block_depth
            self.num_blocks.append(cur_num)
        self.set_channels()    #self.num_filters: 3 blocks with [16,32,64] channels

    def set_channels(self, increase_scale=2):
        self.num_filters=[self.num_channels, self.num_channels]
        for i in range(1, len(self.num_blocks)):
            self.num_filters.append(self.num_filters[i]*increase_scale)

    def get_symbol(self):
        # start network definition
        data = mx.sym.Variable(name='data')
        # first convolution
        data=self.get_conv('g0', data, kout=self.num_filters[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        # different blocks
        data=self.get_group('g1', data, self.num_blocks[0], self.num_filters[0], self.num_filters[1])
        data=self.get_group('g2', data, self.num_blocks[1], self.num_filters[1], self.num_filters[2])
        data=self.get_group('g3', data, self.num_blocks[2], self.num_filters[2], self.num_filters[3])
        # classification layer
        data=self.get_fc('cls', data)
        softmax = mx.sym.SoftmaxOutput(name='softmax', data=data)
        return softmax