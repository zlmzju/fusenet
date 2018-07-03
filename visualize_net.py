import mxnet as mx
import argparse
import visualization as vis

def cal_params(symbol,input_shapes={"data":(1, 3, 32, 32)}):
    """Initialize weight parameters and auxiliary states"""
    arg_shapes, _, _ = symbol.infer_shape(**input_shapes)
    assert(arg_shapes is not None)

    arg_names = symbol.list_arguments()
    input_names = input_shapes.keys()
    param_names = [key for key in arg_names if key not in input_names]

    param_name_shapes = [x for x in zip(arg_names, arg_shapes) if x[0] in param_names]
    import numpy as np
    params_num=0
    for k, s in param_name_shapes:
        params_num+=np.prod(s)
    return '%.4fM'%(params_num/1000000.0)

parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('network', type=str, default='resnet_origin',
                    help = 'the cnn to use, choices = [\'resnet_origin\', \'resnet_plain\', \'resnet_fuse[1,2]\']')
parser.add_argument('depth', type=int, default=50,
                    help = 'depth 29,50,101')
args = parser.parse_args()

# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(1000,args.depth)
print cal_params(net,input_shapes={"data":(1, 3, 224, 224)})
ignore_nodes=["Activation","BatchNorm","Flatten","_MulScalar","Pooling32","SoftmaxOutput"]
dot = vis.plot_network(net, shape={"data":(1, 3, 224, 224)}, 
				node_attrs={"shape":'rect',"fixedsize":'false','fontsize':"72",'fontname':'Helvetica-Bold','ratio':"auto",
				'width':"0", 'height':"0",'len':'0.1', 'margin':"1,0.0"},
                ignore_nodes=ignore_nodes)
dot.graph_attr.update({'rankdir':'BT'})
dot.format = 'jpg'
dot.render('visualize/'+args.network, view=True)