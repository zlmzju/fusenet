# coding: utf-8
# pylint: disable=invalid-name, too-many-locals, fixme
# pylint: disable=too-many-branches, too-many-statements
# pylint: disable=dangerous-default-value
"""Visualization module"""
from __future__ import absolute_import
import matplotlib.cm
import numpy as np
from mxnet.symbol import Symbol
import json
import re
import copy


def _str2tuple(string):
    """convert shape string to list, internal use only

    Parameters
    ----------
    string: str
        shape string

    Returns
    -------
    list of str to represent shape
    """
    return re.findall(r"\d+", string)

def plot_network(symbol, title="plot", shape=None, node_attrs={}):
    """convert symbol to dot object for visualization

    Parameters
    ----------
    title: str
        title of the dot graph
    symbol: Symbol
        symbol to be visualized
    shape: dict
        dict of shapes, str->shape (tuple), given input shapes
    node_attrs: dict
        dict of node's attributes
        for example:
            node_attrs={"shape":"oval","fixedsize":"fasle"}
            means to plot the network in "oval"
    Returns
    ------
    dot: Diagraph
        dot object of symbol
    """
    # todo add shape support
    try:
        from graphviz import Digraph
    except:
        raise ImportError("Draw network requires graphviz library")
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    draw_shape = False
    if shape != None:
        draw_shape = True
        interals = symbol.get_internals()
        _, out_shapes, _ = interals.infer_shape(**shape)
        if out_shapes == None:
            raise ValueError("Input shape is incompete")
        shape_dict = dict(zip(interals.list_outputs(), out_shapes))

    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    heads = set(conf["heads"][0])  # TODO(xxx): check careful
    heads.add(0)
    # default attributes of node
    node_attr = {"shape": "box", "fixedsize": "true",
                 "width": "1.3", "height": "0.8034", "style": "filled"}
    # merge the dict provided by user and the default one
    node_attr.update(node_attrs)
    dot = Digraph(name=title)
    # color map
    cm_origin= ["#8dd3c7", "#ff6666", "#ffff88", "#aaaadd", "#88bbff",
          "#fdb462", "#b3de69", "#fccde5", "#ffa500"]
    cm_left = ["#ddeedd", "#ff5555", "#ffeeb0", "#c0cbeb", "#90c1ff",
          "#ffc4ff", "#c3eeff", "#ffaaff", "#aacbff"]
    # cm
    cm=["#66c2a5","#fc8d62","#8da0cb","#e78ac3",
        "#a6d854","#ffd92f","#e5c494","#b3b3b3"]
    cm.extend(cm_origin)
    #all_colors = matplotlib.cm.rainbow(np.linspace(0, 1, 20))
    # cm=all_colors[:9]
    # cm_left=all_colors[10:]
    # make nodes
    ignore_nodes=["Activation","BatchNorm","Flatten","_MulScalar","Pooling","SoftmaxOutput"]
    conv_num=0
    global_conv_num=0
    left_conv_num=0
    conv_num_dict={}
    left_num_dict={}
    for i in range(len(nodes)):
        node = nodes[i]
        op = node["op"]
        node["name"]=node["name"]+'_%d'%i
        name = node["name"]
        layer_num=0
        # input data
        attr = copy.deepcopy(node_attr)
        label = op
        if op in ignore_nodes:
            continue

        attr['penwidth']='0'
        if op == "null":
            if i in heads:
                label = name[:name.rfind('_')]
                attr["fillcolor"] = cm[0]
                if 'data' in label:
                    label='Input'
                    attr["fillcolor"] = 'white'
            else:
                continue
        elif op == "Convolution":
            layer_num=1
            try:
                group_num=int(name[name.find('g')+1:name.find('g')+2])+1
            except:
                group_num=1
            is_left=True if ('cut' in name) or ('line' in name) or ('short' in name) else False
            cur_color=cm[group_num-2]#cm_left[group_num] if is_left else cm[group_num-2]
            cur_conv_name=name[len('uncouple')+2:name.rfind('_')]
            cur_conv_name=cur_conv_name if 'uncouple' in name else name
            if is_left:
                left_conv_num+=1
                if cur_conv_name not in left_num_dict.keys():
                    left_num_dict[cur_conv_name]=left_conv_num
                conv_num=left_num_dict[cur_conv_name]
            else:
                if cur_conv_name not in conv_num_dict.keys():
                    conv_num_dict[cur_conv_name]=global_conv_num
                    global_conv_num+=1
                conv_num=conv_num_dict[cur_conv_name]
            label = r"Conv%02d, %sx%s/%s, %03s" % (conv_num,_str2tuple(node["param"]["kernel"])[0],
                                                    _str2tuple(node["param"]["kernel"])[1],
                                                    _str2tuple(node["param"]["stride"])[0],
                                                    node["param"]["num_filter"])
            label = r"Conv%02d, %03s"%(conv_num,node["param"]["num_filter"])
            label = r"Conv%02d"%conv_num
            if 'uncouple' not in name:
                label = r"Conv"
            if 'nonshare' in name or 'uncouple' in name:
                label = r"C%02d"%conv_num
            label = "L" if is_left else "R"
            if conv_num==0:
                label="C"
            label += r"%02d"%conv_num
            if '_l' in name and 'uncouple' not in name:
                attr["shape"]="none"
                label="C"
            attr["fillcolor"] = cur_color
        elif op == "FullyConnected":
            layer_num=1
            label = r"FC,%s" % node["param"]["num_hidden"]
            label = r" FC "
            attr["fillcolor"] = '#a6d854'
            # attr['margin']='0.2,0.1'
        elif op == "BatchNorm":
            label= r"BN"
            attr["fillcolor"] = cm[3]
        elif op == "Activation" or op == "LeakyReLU":
            label = r"%s" % (node["param"]["act_type"])
            attr["fillcolor"] = cm[8]
        elif op == "Pooling":
            # label = r"Pooling,%s,%sx%s/%s" % (node["param"]["pool_type"],
            #                                     _str2tuple(node["param"]["kernel"])[0],
            #                                     _str2tuple(node["param"]["kernel"])[1],
            #                                     _str2tuple(node["param"]["stride"])[0])
            label=r"Pooling"
            attr["fillcolor"] = cm[0]
        elif op == "Concat" or op == "Flatten" or op == "Reshape":
            attr["fillcolor"] = cm[5]
        elif op == "Softmax":
            attr["fillcolor"] = cm[6]
        elif op == "SoftmaxOutput":
            label=r"Softmax"
            attr["fillcolor"] = cm[7]
        elif op == "_Plus" or op == "ElementWiseSum" :
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_op = input_node["op"]
                if input_op=="_Plus":   #duplicated node name will be merged by graphviz
                    node["name"]=input_node["name"]
                    name=node["name"]
                    node["inputs"].remove(item)
            label = r"+"
            attr['shape']='circle'
            # attr['shape']='point'
            attr['margin']="0,0"
            attr['width']="0.1"
            attr['height']="0.1"
            attr["fillcolor"] = 'white'
            attr['penwidth']='2'
        else:
            attr["fillcolor"] = cm[8]
        attr['pencolor']='#262626'
        dot.node(name=name, label=label, **attr)

    # add edges
    for i in range(len(nodes)):
        node = nodes[i]
        op = node["op"]
        name = node["name"]
        if op == "null":
            continue
        elif op in ignore_nodes:
            continue
        else:
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                while True:
                    if input_node["op"] in ignore_nodes:
                        cur_input = input_node["inputs"][0]
                        input_node = nodes[cur_input[0]]
                        input_name = input_node["name"]
                    else:
                        break
                if input_node["op"] != "null" or item[0] in heads:
                    attr ={"dir": "back"}
                    # add shapes
                    if draw_shape:
                        if input_node["op"] != "null":
                            key = input_name[:input_name.rfind('_')] + "_output"
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            # attr["label"] = label
                            # attr['fontsize']="46"
                        else:
                            key = input_name[:input_name.rfind('_')]
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            #attr["label"] = label
                    dot.edge(tail_name=name, head_name=input_name, minlen='1.0',**attr)
    return dot


