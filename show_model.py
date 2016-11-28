'''
Show the defined network symbols.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
import argparse
import os
import sys
import logging
import numpy as np
import options
import visualization as vis

def show_net(net, name, data_size=224):
    #only show conv and fc for simplicity, ignore the following layers
    ignore_nodes=["Activation","BatchNorm","Flatten","_MulScalar","Pooling","SoftmaxOutput"]
    #figure style
    node_attrs={"shape":'rect',"fixedsize":'false','fontsize':"54",'fontname':'Arial','ratio':"auto",
                    'width':"0", 'height':"0",'len':'0.1', 'margin':"0.3,0.1", 'penwidth':'3.0'}
    input_shape={"data":(1, 3, data_size, data_size)}
    #show
    dot = vis.plot_network(net, input_shape, node_attrs, ignore_nodes)
    dot.graph_attr.update({'rankdir':'BT'})
    dot.format = 'png' #'png'
    file_name='visualize/'+name
    dot.render(file_name, view=True)
    return file_name+'.'+dot.format

def main(argv):
    args = options.get_args(argv,parse=False)
    net = options.get_network(args)
    args.network+='_d%d'%args.depth
    file_name=show_net(net,args.network,args.data_shape)


if __name__ == '__main__':
    main(sys.argv[1:])