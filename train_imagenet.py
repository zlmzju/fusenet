'''
Reproducing https://github.com/gcr/torch-residual-networks

References:

Kaimin He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
'''
import mxnet as mx
import argparse
import os
import logging
import time
import numpy as np

parser = argparse.ArgumentParser(description='train an image classifer on ImageNet')
parser.add_argument('--network', type=str, default='symbol_cross',
                    help = 'the cnn to use, choices = [\'symbol_cross\', \'symbol_resnet\']')
parser.add_argument('--data-dir', type=str, default=r'../imagenet/',
                    help='the input data directory')
parser.add_argument('--gpus', type=str, default='0,1,2,3',
                    help='the gpus will be used, e.g "0,1,2,3,4,5,6,7"')
parser.add_argument('--batch-size', type=int, default=256,
                    help='the batch size')
parser.add_argument('--model-prefix', type=str, default='./imagenet/',
                    help='the prefix of the model to load/save')
parser.add_argument('--load-model', type=str, 
                    help='the name of the model to load')
parser.add_argument('--num-epochs', type=int, default=120,
                    help='the number of training epochs')
parser.add_argument('--log-file', type=str, default='log.txt', help='the name of log file')
parser.add_argument('--log-dir', type=str, default='./imagenet/', help='directory of the log file')
parser.add_argument('--lr', type=float, default=0.1,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-epoch-step', type=int, default=30,
                    help='the number of epoch to factor the lr, could be 10')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='device',
                    help='the kvstore type')
parser.add_argument('--num-examples', type=int, default=1281167,
                    help='the number of training examples')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='the number of classes')
parser.add_argument('--train-dataset', type=str, default="imagenet_train.rec",
                    help='train dataset name')
parser.add_argument('--val-dataset', type=str, default="imagenet_val.rec",
                    help="validation dataset name")
parser.add_argument('--data-shape', type=int, default=224,
                    help='set image\'s shape')
parser.add_argument('--aug-type', type=int, default=1,
                    help='augmentation type')
parser.add_argument('--rand_seed', type=int, 
                    help='random seed for initialization')
parser.add_argument('--fb-mean', type=bool, default=False,
                    help='using my own mean values or fbs')
args = parser.parse_args()

# network
import importlib
import sys
net = importlib.import_module(args.network).get_symbol(args.num_classes)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"]='1'
if args.rand_seed is None:
    import time     
    args.rand_seed=int(time.time()) #different random init for serveral runs
mx.random.seed(args.rand_seed)      #cudnn conv backward is non-deterministic
exp_name=args.log_file[:args.log_file.rfind('.')]
args.network='net_'+args.network+'/'+args.network+'_'+exp_name+'/'+args.network

def get_iterator(args, kv):
    kargs = dict(
        data_shape=(3, args.data_shape, args.data_shape),
        #zscore
        mean_r=123.675 if args.fb_mean else 123.370,
        mean_g=116.280 if args.fb_mean else 112.757,
        mean_b= 103.530 if args.fb_mean else 99.406,
        fill_value_r=124 if args.fb_mean else 123,
        fill_value_g=116 if args.fb_mean else 113,
        fill_value_b=104 if args.fb_mean else 99,
        scale_r=(1.0 / 58.395) if args.fb_mean else (1.0 / 68.998),
        scale_g=(1.0 / 57.12) if args.fb_mean else (1.0 / 66.093),
        scale_b=(1.0 / 57.375) if args.fb_mean else (1.0 / 68.292),
    )
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, args.train_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        #data augmentation: affine transformation->random crop
        rand_crop=True,
        rand_mirror=True,
        min_random_scale=0.533,  #480*0.533=255.84
        max_random_scale=1.0 if args.aug_type>0 else 0.533,
        min_img_size=256,		#255.84->256; [256,480]
        max_aspect_ratio=0.25 if args.aug_type>0 else 0, #aspect [0.75,1.25]
        #random color jitter
        random_h=36 if args.aug_type>0 else 0,
        random_s=50 if args.aug_type>0 else 0,
        random_l=50 if args.aug_type>0 else 0,
        # #others
        inter_method=9, #auto select resize method: bilinear, cubic, etc.
        num_parts=kv.num_workers,
        part_index=kv.rank,
        # Preprocessing thread number
        preprocess_threads=8,
        **kargs
    )
    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, args.val_dataset),
        shuffle=False,
        rand_crop=False,    #center crop
        rand_mirror=False,        
        min_random_scale=0.533,  #480*0.533=255.84
        max_random_scale=0.533,   #480*0.533=255.84
        min_img_size=256,		#255.84->256
        inter_method=9, #auto
        batch_size=args.batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        # Preprocessing thread number
        preprocess_threads=8,
        **kargs
    )
    return (train, val)

class Init(mx.init.Xavier):
    def __init__(self,rnd_type="uniform", factor_type="avg", magnitude=3):
        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.ndarray.NDArray):
            raise TypeError('arr must be NDArray')
        if name.endswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)

        
class Scheduler(mx.lr_scheduler.MultiFactorScheduler):

    def __init__(self, epoch_step, factor, epoch_size):
        super(Scheduler, self).__init__(
            step=[epoch_size * s for s in epoch_step],
            factor=factor
        )


@mx.optimizer.Optimizer.register
class Nesterov(mx.optimizer.NAG):
    def set_wd_mult(self, args_wd_mult):
        self.wd_mult = {}
        for n in self.idx2name.values():
            if not (
                n.endswith('_weight')
                or n.endswith('_bias')
                or n.endswith('_gamma')
                or n.endswith('_beta')
                or n.endswith('weightfuse')
            ):
                self.wd_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.attr_dict()
            for k, v in attr.items():
                if k.endswith('_wd_mult'):
                    self.wd_mult[k[:-len('_wd_mult')]] = float(v)
        self.wd_mult.update(args_wd_mult)

class InfoCallback(mx.callback.Speedometer):
    """Calculate training speed in frequent

    Parameters
    ----------
    batch_size: int
        batch_size of data
    frequent: int
        calculation frequent
    """
    def __init__(self, batch_size, frequent=50):
        mx.callback.Speedometer.__init__(self, batch_size, frequent)
        self.total_top1=0.0
        self.total_top5=0.0
        self.total_loss=0.0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    param.eval_metric.reset()
                    log_info='Epoch[%d] Batch [%d]\tSpeed: %.0f'%(param.epoch, count, speed)
                    for name, value in name_value:
                    	value=value if not np.isinf(value) else 10.0 #log(0)=inf
                        if name=='cross-entropy':
                            self.total_loss+=1.0*value*self.frequent
                            log_info=log_info+'\tloss: %.4f(%.4f)'%(value,self.total_loss/count)
                        elif name=='accuracy':
                            self.total_top1+=1.0*value*self.frequent
                            log_info=log_info+'\ttop1: %.4f(%.4f)'%(100.0*value,100.0*self.total_top1/count)
                        elif 'top_k' in name:
                            self.total_top5+=1.0*value*self.frequent
                            log_info=log_info+'\ttop5: %.4f(%.4f)'%(100.0*value,100.0*self.total_top5/count)
                    logging.info(log_info)
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
            self.total_top1=0.0
            self.total_top5=0.0
            self.total_loss=0.0
            
def fit(args, network, data_loader, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.network+'.txt'
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        log_dir=os.path.dirname(log_file_full_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # import shutil
        # shutil.copy('train_imagenet.py', os.path.join(log_dir, 'training.py'))
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    assert model_prefix is not None
    model_prefix +=args.network
    print model_prefix
    model_args = {}
    start_epoch_step=args.lr_epoch_step
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params': tmp.arg_params,
                      'aux_params': tmp.aux_params,
                      'begin_epoch': args.load_epoch}
        stage=int(args.load_epoch/args.lr_epoch_step)
        args.lr*=args.lr_factor**stage
        start_epoch_step=(1+stage)*args.lr_epoch_step-args.load_epoch
    if args.load_model is not None:
        save_dict = mx.nd.load('%s.params' % (args.model_prefix+args.load_model))
        tmp_arg_params = {}
        tmp_aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                tmp_arg_params[name] = v
            if tp == 'aux':
                tmp_aux_params[name] = v
        model_args = {'arg_params': tmp_arg_params,
                      'aux_params': tmp_aux_params}

    # save model
    checkpoint = mx.callback.do_checkpoint(model_prefix,5)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(i) for i in range(len(args.gpus.split(',')))]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None
        
    epoch_step=range(start_epoch_step,args.num_epochs,args.lr_epoch_step)
    model_args['lr_scheduler'] = Scheduler(epoch_step=epoch_step,
                                                factor=args.lr_factor, epoch_size=epoch_size)   

    logger.info('training parameters: lr=%f, epoch_size=%d, epoch_step=%s',args.lr,epoch_size,epoch_step)

    model = mx.model.FeedForward(
        ctx=devs,
        symbol=network,
        num_epoch=args.num_epochs,
        learning_rate=args.lr,
        momentum=0.9,
        wd=0.0001,
        optimizer='Nesterov',
        # Note we initialize BatchNorm beta and gamma as that in
        # https://github.com/facebook/fb.resnet.torch/
        # i.e. constant 0 and 1, rather than
        # https://github.com/gcr/torch-residual-networks/blob/master/residual-layers.lua
        # FC layer is initialized as that in torch default
        # https://github.com/torch/nn/blob/master/Linear.lua
        initializer=mx.init.Mixed(
            ['.*fc.*', '.*'],
            [mx.init.Xavier(rnd_type='uniform',  factor_type='in', magnitude=1),
             Init(rnd_type='gaussian', factor_type='in', magnitude=2)]
        ),
        #lr_scheduler=Scheduler(epoch_step=[30, 60, 90, 120, 150, 180], factor=0.1, epoch_size=epoch_size),
        **model_args)

    eval_metrics = ['ce','accuracy']
    ## TopKAccuracy only allows top_k > 1
    for top_k in [5]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(InfoCallback(args.batch_size, 10))

    model.fit(
        X=train,
        eval_data=val,
        eval_metric=eval_metrics,
        kvstore=kv,
        batch_end_callback=batch_end_callback,
        epoch_end_callback=checkpoint
    )

# train
fit(args, net, get_iterator)
