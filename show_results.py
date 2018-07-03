import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt 

def read_file(file_name):
    file=open(file_name)
    contents=file.readlines()
    train_result_dict={}
    val_result_dict={}
    for oneline in contents:
        array=oneline.split()
        if len(array)<4:
            continue
        if len(array)<6:
            epoch=int(array[3].replace(']','').split('[')[-1])+1
            if not val_result_dict.has_key(epoch):
                val_result_dict[epoch]={}
                val_result_dict[epoch]['loss']=[]
                val_result_dict[epoch]['top1']=[]
                val_result_dict[epoch]['top5']=[]
            key=array[4][:14]
            if key=='Validation-cro':
                val_result_dict[epoch]['loss'].append(float(array[4].split('=')[-1]))
            elif key=='Validation-acc':
                val_result_dict[epoch]['top1'].append(float(array[4].split('=')[-1]))
            elif key=='Validation-top':
                val_result_dict[epoch]['top5'].append(float(array[4].split('=')[-1]))
        elif array[5]=='[5000]':
            epoch=int(array[3].replace(']','').split('[')[-1])+1
            if not val_result_dict.has_key(epoch):
                train_result_dict[epoch]={}
                train_result_dict[epoch]['loss']=[]
                train_result_dict[epoch]['top1']=[]
                train_result_dict[epoch]['top5']=[]
            scale= 0.01 if float(array[-1].replace(')','').split('(')[-1])>1 else 1
            train_result_dict[epoch]['loss'].append(float(array[9].replace(')','').split('(')[-1]))
            train_result_dict[epoch]['top1'].append(scale*float(array[11].replace(')','').split('(')[-1]))
            train_result_dict[epoch]['top5'].append(scale*float(array[-1].replace(')','').split('(')[-1]))
    return train_result_dict,val_result_dict

def get_paper(style='acc'):
    x=[1,5,10,15,20,25,30,
       31,35,40,45,50,55,60,
       61,65,70,75,80,85,90]
    top1_val_coarse=[89.3,46.5,45,44.3,42.8,42,41.7,
              32, 29, 29,29.2,29,28.8,29,
              26.2,26.1,26,25.8,25.6,25.0,24.7];
    top1_train_coarse=[95,55,50,48,47,46.5,46,
              34,32,30.5,30,30,30,30,
              24,23.5,22.8,21,20.5,20.3,20.1];
    xvals = np.linspace(1, 90, 90)
    top1_val_fit = np.interp(xvals, x, top1_val_coarse)
    top1_train_fit = np.interp(xvals, x, top1_train_coarse)
    #return 
    bias=0.0
    scale=1.0
    best=np.argmin
    if style=='acc':
        bias=100.0
        scale*=-1.0
        best=np.argmax
    top1_val=[bias+scale*value for value in top1_val_fit]
    top5_val=[bias+scale*7.8]*len(top1_val)
    top1_train=[bias+scale*value for value in top1_train_fit]
    top5_train=[-1.0]*len(top1_train)
    return top1_val,top5_val,top1_train,top5_train,best

def print_result(result_dict,prefix,style='acc'):
    top1=[]
    top5=[]
    loss=[]
    bias=0.0
    scale=100.0
    best=np.argmax
    if style=='err':
        bias=100.0
        scale=-100.0
        best=np.argmin
    length=len(result_dict)
    total_epochs=100
    length=total_epochs if length>total_epochs else length
    for epoch in range(1,length+1):
        res=result_dict[epoch]
        top1.append(bias+scale*res['top1'][-1])
        top5.append(bias+scale*res['top5'][-1])
        loss.append(res['loss'][-1])
        # print(' * %s epoch # %02d     top1: %7.3f  top5: %7.3f   loss: %7.3f'%\
        #                         (prefix, epoch,top1[epoch-1],top5[epoch-1],loss[epoch-1]))
    return top1,top5,loss,best

def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--net",
        default='all',
        help="Network to show: plain, origin, fuse[1-3], and all." +
             "For example: 'python script.py fuse3'."
    )
    parser.add_argument(
        "--style",
        default='acc',
        help="Style to show the accuracy: acc or err (default: acc)." +
             "For example: accuracy=100.0%% for 'acc' and err=0.0%% for 'err'."
    )
    parser.add_argument(
        "--dir",
        default='snapshot',
        help="Directory of the result txt file (default: snapshot)."
    )
    parser.add_argument(
        "--plot",
        default='val',
        help="plot train or val, or both of them (default: all)."
    )
    args = parser.parse_args()
    #whole
    if '50_4' in args.net:
        networks=['dfn-mr_50_4gpu','resnet_50_4gpu']
    elif '50' in args.net:
        networks=['dfn-mr_50_8gpu','resnet_50_8gpu']
    elif '101' in args.net:
        networks=['dfn-mr_101_8gpu','resnet_101_8gpu']
        #'dfn-mr_50_8gpu','resnet_50_8gpu','origin_resnet_50'
        #'dfn-mr_50_4gpu','resnet_50_4gpu','origin_resnet_50'
        #'dfn-mr_101_8gpu','resnet_101_8gpu','origin_resnet_101']
    else:
        networks=['dfn-mr_101_8gpu','resnet_101_8gpu']
    #figure params
    colors=['black','blue','orange','green','red','cyan','pink']
    plt.figure()
    #one net
    print args.net
    import glob
    all_logs=glob.glob('snapshot/*/*/*.txt')
    for idx in range(len(networks)):
        net=networks[idx]
        if 'paper' not in net:
            for log_name in all_logs:
                if net in log_name:
                    file_name=log_name
            print file_name
            train_res,val_res=read_file(file_name)
        else:
            top1_val,top5_val,top1_train,top5_train,best=get_paper(args.style)
        if 'middle' in net:
            net=net.replace('middle','dfn-mr')
        title='(solid lines: 1-crop val error; dashed lines: training error)'
        if args.plot=='train' or args.plot=='all':
            if net!='paper':
                top1,top5,loss,best=print_result(train_res,'Finished',args.style)   
            report_idx=-1#best(top1)
            plot1=plt.plot(range(1,len(top1)+1),top1,color=colors[idx], linestyle=':', linewidth=2.0, alpha=1 if args.plot=='train' else 0.5, 
                marker='o' if args.plot=='train' else None,
                label='%6s top1:%7.2f%% top5:%7.2f%%'%(net,top1[report_idx],top5[report_idx]) if args.plot=='train' else None)

        if args.plot=='val' or args.plot=='all':
            if net!='paper':
                top1,top5,loss,best=print_result(val_res,'Finished',args.style)        
            elif net=='paper':
                top1=top1_val
                top5=top5_val
            if args.plot=='val':
                title='(validation error)'
            report_idx=-1#best(top1)
            plot2=plt.plot(range(1,len(top1)+1),top1,color=colors[idx], linestyle='-', 
                marker='o' if args.plot=='val' else None,linewidth=2.0,
                label='%6s (%d) top1:%7.2f%% top5:%7.2f%%'%(net,len(top1),top1[report_idx],top5[report_idx]))
    #other figure style
    plt.ylim([15,90])
    plt.xticks(np.arange(0, 50, 10))
    #title
    plt.title('Performance of different architectures on ImageNet %s'%(title),fontsize=28)
    plt.ylabel('top1 error',fontsize=24)
    plt.xlabel('epoch',fontsize=24)   
    plt.legend(prop={'size':24,'family':'monospace'})
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
