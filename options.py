'''
Training options on Cifar10, Cifar100, and SVHN.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
import argparse
import os
import logging
import numpy as np
import utility

def get_network(args):
    import importlib
    import sys
    sys.path.insert(0,'network')
    net_module= importlib.import_module(args.network)
    network= net_module.get_symbol(args.num_classes, args.depth, args.widen_factor)
    data_shape=(1, 3, args.data_shape, args.data_shape)
    logging.warning('network parameters: %s',utility.cal_params(network,input_shapes={"data":data_shape}))
    return network
    
#parse the arguments
def parse_args(args, parse=True):
    __dataset_args(args)
    if parse:
        __logging_args(args)
        __training_args(args)

def __dataset_args(args):
    if args.dataset=='cifar10':
        args.mean_rgb=[125.307, 122.950, 113.865]
        args.std_rgb=[62.993, 62.089, 66.705]
        args.test_batch_size=400 if args.batch_size<400 else args.batch_size #mxnet issues: test_batch>=train_batch
    elif args.dataset=='cifar100':
        args.num_classes=100
        args.mean_rgb=[129.304, 124.070, 112.434]
        args.std_rgb=[68.170, 65.392, 70.418]
    elif args.dataset=='svhn':
        args.num_epochs=40 if args.num_epochs>100 else args.num_epochs  #default is 40 if not specified
        args.num_examples=604388
        args.mean_rgb=[111.609, 113.161, 120.565]
        args.std_rgb=[50.498, 51.259, 50.244]
        args.aug_type=0 #no data augmentation
        args.test_batch_size=280 if args.batch_size<280 else args.batch_size #26032 test images
    elif args.dataset=='imagenet': #TODO: imagenet training
        args.num_epochs=100
        args.data_shape=224
        args.num_examples=1281167
        args.num_classes=1000 
        args.mean_rgb=[123.370, 112.757, 99.406] #calculated on the resized training data (short side = 480)
        args.std_rgb=[68.998, 66.093, 68.292]
        args.aug_type=2 #extreme data augmentation
        args.test_batch_size=200 if args.batch_size<200 else args.batch_size
    if args.data_dir is None:
        args.data_dir='../../../dataset/'+args.dataset+'/'

def __logging_args(args):
    #make directories
    args.log_dir+=args.dataset+'_noaug/' if args.aug_type==0 else args.dataset+'/'
    logfile_name='%s_d%dw%d'%(args.network,args.depth,args.widen_factor)
    #name of logging text file
    if args.model_prefix is None:
        args.model_prefix=args.log_dir+args.network+'/'
        logfile_name+='_exp' if args.exp_name is None else '_'+args.exp_name
        random_idx=1
        while os.path.isfile(args.model_prefix+logfile_name+str(random_idx)+'.txt'):
            random_idx+=1
        logfile_name+=str(random_idx)
    #model related
    if args.checkpoint_epochs is None: #if num_epochs=400, then we will save the model every 50 epochs
        args.checkpoint_epochs=args.num_epochs/8
    if args.rand_seed is None:
        import time     
        args.rand_seed=int(time.time()) #different random init for serveral runs
    mx.random.seed(args.rand_seed)      #cudnn conv backward is non-deterministic
    #logging
    log_file_full_name = args.model_prefix+logfile_name+'.txt'
    args.model_prefix+='weights/'+logfile_name+'/'
    utility.mkdir(args.model_prefix)
    args.model_prefix+=logfile_name
    head = '%(asctime)-15s %(message)s'
    logger = logging.getLogger()
    map(logger.removeHandler, logger.handlers[:]) #reset
    handler = logging.FileHandler(log_file_full_name)
    formatter = logging.Formatter(head)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('%s',log_file_full_name)
    logger.info('start with arguments %s', args)

def __training_args(args):
    if args.lr_steps is None:
        args.lr_steps=[args.num_epochs*1/2,args.num_epochs*3/4,args.num_epochs*7/8]
    else:
        args.lr_steps=[int(v) for v in args.lr_steps.split(',')]
    if args.load_epoch is not None:
        tmp = mx.model.FeedForward.load(args.model_prefix, args.load_epoch)
        args.model_args = { 'arg_params': tmp.arg_params,
                            'aux_params': tmp.aux_params,
                            'begin_epoch': args.load_epoch}
        origin_step=args.lr_steps[:]
        args.lr_steps=[]
        for tmp_lr in origin_step:
            if tmp_lr<=args.load_epoch:
                args.lr*=args.lr_factor
            else:
                args.lr_steps.append(tmp_lr-args.load_epoch)
    #gpus
    logging.info("Using gpus %s from:\n%s",args.gpus, os.popen("nvidia-smi -L").read())
    logging.info('training strategy: lr=%f, step=%s',args.lr,args.lr_steps)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

def get_args(argv,parse=True):
    parser = argparse.ArgumentParser(description='train an deep fusion network: "python train_model.py cifar10 resnet 56"')
    #network parameters
    parser.add_argument('--network', type=str, default='cross', help = 'network name')
    parser.add_argument('--depth', type=int, default=62, help = 'depth of the corresponding plain network (if 62 for CrossNet, the actual depth is 32)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100','svhn', 'imagenet'], help='dataset name')
    parser.add_argument('--widen-factor', type=int, default=1, help='channel number based on 16')
    #for logging experiments
    parser.add_argument('--log-dir', type=str, default='./snapshot/', help='directory of the log file')
    parser.add_argument('--exp-name', type=str, help='experiment description for logging same network')
    parser.add_argument('--checkpoint-epochs', type=int, help='save the model every N epochs')  
    parser.add_argument('--log-iters', type=int, default=50, help='logging info every N iterations')  
    #training strategy
    parser.add_argument('--gpus', type=str, default='0,1', help='the gpus will be used, e.g., --gpus=0,1')
    parser.add_argument('--batch-size', type=int, default=64, help='the training batch size')
    parser.add_argument('--test-batch-size', type=int, default=400, help='the testing batch size')
    parser.add_argument('--num-epochs', type=int, default=400, help='the number of training epochs')  
    parser.add_argument('--rand-seed', type=int, help='None for different random seed for each run')    
    #learning rate
    parser.add_argument('--lr', type=float, default=0.1, help='the initial learning rate')
    parser.add_argument('--lr-factor', type=float, default=0.1, help='reduce the lr by a factor')
    parser.add_argument('--lr-steps', type=str, help='reduce the lr by a factor e.g., --lr-steps=100,150')
    #dataset locations
    parser.add_argument('--data-dir', type=str, help='the input data directory')
    parser.add_argument('--train-dataset', type=str, default="train.rec", help='train dataset name')
    parser.add_argument('--val-dataset', type=str, default="test.rec", help="validation dataset name")
    #dataset settings
    parser.add_argument('--num-examples', type=int, default=50000, help='the number of training examples')
    parser.add_argument('--num-classes', type=int, default=10, help='the number of classes')
    parser.add_argument('--data-shape', type=int, default=32, help='set image\'s shape')
    parser.add_argument('--mean-rgb', type=list, default=[127, 127, 127], help='image mean values')
    parser.add_argument('--std-rgb', type=list, default=[60, 60, 60], help='image std values')
    parser.add_argument('--aug-type', type=int, default=1, choices=[0,1,2], help='data augmentation type: 0 (no aug), 1 (+), 2 (++)')
    #retrain
    parser.add_argument('--model-prefix', type=str, help='the prefix of the model to load')
    parser.add_argument('--load-epoch', type=int, help="load the model on an epoch using the model-prefix")
    parser.add_argument('--model-args', type=dict, default={}, help="internal usage for loading model")
    #mxnet for multi-gpu update
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type in mxnet')
    args = parser.parse_args(argv)
    #parse arguments
    parse_args(args,parse)
    return args