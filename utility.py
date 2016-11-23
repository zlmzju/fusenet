'''
Utility functions for training.
Contact: Liming Zhao (zlmzju@gmail.com)
'''
import mxnet as mx
import numpy as np
import time
import logging

#utility functions
def mkdir(dirname,clean=False):
    import os
    if clean and os.path.exists(dirname):
        import shutil
        shutil.rmtree(dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

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

class Scheduler(mx.lr_scheduler.MultiFactorScheduler):
    def __init__(self, epoch_step, factor, epoch_size):
        super(Scheduler, self).__init__(
            step=[epoch_size * s for s in epoch_step],
            factor=factor
        )

@mx.optimizer.Optimizer.register
class Nesterov(mx.optimizer.NAG):
    #same with torch implementation
    def set_wd_mult(self, args_wd_mult):
        self.wd_mult = {}
        for n in self.idx2name.values():
            if not ( n.endswith('_weight') or n.endswith('_bias')
                    or n.endswith('_gamma') or n.endswith('_beta')
            ):
                self.wd_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.list_attr(recursive=True)
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