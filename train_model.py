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
import utility

def get_iterator(args, kv):
    base_args=dict(
        num_parts=kv.num_workers,
        part_index=kv.rank,
        # Image normalization
        data_shape=(3, args.data_shape, args.data_shape),
        # subtract mean and divide std
        mean_r=args.mean_rgb[0],
        mean_g=args.mean_rgb[1],
        mean_b=args.mean_rgb[2],
        fill_value_r=int(round(args.mean_rgb[0])),
        fill_value_g=int(round(args.mean_rgb[1])),
        fill_value_b=int(round(args.mean_rgb[2])),
        scale_r=1.0 / args.std_rgb[0],
        scale_g=1.0 / args.std_rgb[1],
        scale_b=1.0 / args.std_rgb[2],
    )
    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir+args.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        #image augmentation
        pad=4 if args.aug_type==1 else 0,
        rand_crop=True if args.aug_type!=0 else False,
        rand_mirror=True if args.aug_type!=0 else False,
        **base_args #base arguments for mxnet
    )
    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir+args.val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        rand_crop=False,
        rand_mirror=False,
        **base_args #base arguments for mxnet
    )
    return (train, val)

def train(args):
    network=options.get_network(args)
    #device
    kv = mx.kvstore.create(args.kv_store)    
    devs = [mx.gpu(i) for i in range(len(args.gpus.split(',')))]
    #training data
    (train, val)=get_iterator(args, kv)
    #model
    model = mx.model.FeedForward(
        ctx=devs,
        symbol=network,
        num_epoch=args.num_epochs,
        learning_rate=args.lr,
        momentum=0.9,
        wd=0.0001,
        optimizer='Nesterov', #'nag',
        initializer=mx.init.Mixed(['.*fc.*', '.*'],
            [mx.init.Xavier(rnd_type='uniform', factor_type='in', magnitude=1),
             mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)]),
        lr_scheduler=utility.Scheduler(epoch_step=args.lr_steps, factor=args.lr_factor, epoch_size=args.num_examples / args.batch_size),
        **args.model_args #for retrain
    )
    model.fit(
        X=train,
        eval_data=val,
        eval_metric=['ce','acc'] if args.dataset!='imagenet' else ['ce','acc',mx.metric.create('top_k_accuracy',top_k=5)],
        kvstore=kv,
        batch_end_callback=utility.InfoCallback(args.batch_size, 50),
        epoch_end_callback=mx.callback.do_checkpoint(args.model_prefix,args.num_epochs/8),
    )

def main():
    args = options.get_args()
    train(args)

if __name__ == '__main__':
    main()