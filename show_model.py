'''
Train model on Cifar10, Cifar100, and SVHN.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
import argparse
import os
import logging
import numpy as np
import options
import visualization as vis

args = options.get_args(parse=False)
net = options.get_network(args)
args.network+='_d%d'%args.depth

dot = vis.plot_network(net, shape={"data":(1, 3, args.data_shape, args.data_shape)}, 
				node_attrs={"shape":'rect',"fixedsize":'false','fontsize':"54",'fontname':'Arial','ratio':"auto",
				'width':"0", 'height':"0",'len':'0.1', 'margin':"0.3,0.1", 'penwidth':'3.0'})
dot.graph_attr.update({'rankdir':'BT'})
dot.format = 'png' #'png'
dot.render('visualize/'+args.network, view=True)