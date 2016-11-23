import sys
import glob
import numpy as np
import matplotlib.pyplot as plt 

def read_file(file_name):
    file=open(file_name)
    contents=file.readlines()
    train_result_dict={}
    val_result_dict={}

    for line in contents:
        if 'Epoch' not in line:
            continue
        epoch_str=line[line.find('Epoch')+5:]
        batch_str=line[line.find('Batch')+5:]
        epoch=int(epoch_str[epoch_str.find('[')+1:epoch_str.find(']')])+1
        if 'Validation' in line:
            if not val_result_dict.has_key(epoch):
                val_result_dict[epoch]={}
                val_result_dict[epoch]['loss']=[]
                val_result_dict[epoch]['top1']=[]
                val_result_dict[epoch]['top5']=[]
            key=line[line.find('Validation'):]
            value=key[key.find('=')+1:]
            if 'Validation-cro' in key:
                val_result_dict[epoch]['loss'].append(float(value))
            elif 'Validation-acc' in key:
                val_result_dict[epoch]['top1'].append(float(value))
            elif 'Validation-top' in key:
                val_result_dict[epoch]['top5'].append(float(value))
        elif '[750]' in batch_str:
            if not val_result_dict.has_key(epoch):
                train_result_dict[epoch]={}
                train_result_dict[epoch]['loss']=[]
                train_result_dict[epoch]['top1']=[]
                train_result_dict[epoch]['top5']=[]
            loss=line[line.find('loss')+4:]
            loss=loss[loss.find('(')+1:loss.find(')')]
            top1=line[line.find('top1')+4:]
            top1=top1[top1.find('(')+1:top1.find(')')]   
            top5=line[line.find('top5')+4:]
            top5=top5[top5.find('(')+1:top5.find(')')]   
            try:
                loss=float(loss)
                top1=float(top1)
                top5=float(top5)
            except:
                loss=loss if isinstance(loss,float) else -1
                top1=top1 if isinstance(top1,float) else -1
                top5=top5 if isinstance(top5,float) else -1
            scale= 0.01 if top1>1 else 1
            train_result_dict[epoch]['loss'].append(loss)
            train_result_dict[epoch]['top1'].append(scale*top1)
            train_result_dict[epoch]['top5'].append(scale*top5)
        # elif array[5]=='arguments':
            #print array[6:]
    return train_result_dict,val_result_dict

def print_result(result_dict,style='acc'):
    top1=[]
    top5=[]
    loss=[]
    bias=0.0
    scale=100.0
    best=max
    if style=='err':
        bias=100.0
        scale=-100.0
        best=min
    for epoch in range(1,len(result_dict)+1):
        res=result_dict[epoch]
        top1.append(bias+scale*res['top1'][-1])
        if 'top5' in res.keys() and len(res['top5'])>0:
            top5.append(bias+scale*res['top5'][-1])
        loss.append(res['loss'][-1])
    return top1,top5,loss,best

def sort_results(t_dict_list,test_idx=-1):
    t_res=[]
    t_list=[]
    loss_list=[]
    for t_dict in t_dict_list:
        t_top1,_,t_loss,_=print_result(t_dict,style='acc')
        t_list.append(t_top1)
        loss_list.append(t_loss)
        t_res.append(t_top1[-1])
    if test_idx==-1:
        mid=int( (len(t_res)-1)/2 )
        index=np.argsort(t_res)[mid]
    else:
        index=test_idx
    t_str='%6.3f%% (%6.3f +/- %6.3f, %6.3f)'%(t_res[index],np.mean(t_res),np.std(t_res), np.max(t_res))
    top_curve=t_list[index]
    loss_curve=loss_list[index]
    return index,t_str,top_curve, loss_curve

def net_results(exp_name):
    file_names= glob.glob('%s*.txt'%(exp_name))
    train_list=[]
    test_list=[]
    valid_names=[]
    for file_name in file_names:
        train_dict,test_dict=read_file(file_name)
        if len(train_dict)%40!=0:
            continue
        valid_names.append(file_name)
        train_list.append(train_dict)
        test_list.append(test_dict)
#     print '\n',len(valid_names),valid_names
    if len(test_list)<1:
        return None,None,None,0
    index,test_str,test_top1,test_loss=sort_results(test_list)
    _,train_str,train_top1,train_loss=sort_results(train_list,index)
    train_tuple=(train_str,train_top1,train_loss)
    test_tuple=(test_str,test_top1,test_loss)
    return train_tuple,test_tuple,valid_names[index],len(test_list)

def read_result(result_filename):
    with open(result_filename,'r') as file:
        contents=file.readlines()
        print contents[0][:-1]

def process_exps(snapshot_dir, exp_names,override=False):
    for exp_name in exp_names:
        #original file
        result_filename=snapshot_dir+'_result/'+exp_name[exp_name.rfind('/')+1:]+'.txt'
        import os
        if not override and os.path.isfile(result_filename):
            read_result(result_filename)
            continue #it has been processed
        #read results
        train_tuple,test_tuple,file_name,run_num=net_results(exp_name)
        if run_num<1:
        	continue
        train_str,train_top1,train_loss=train_tuple
        test_str,test_top1,test_loss=test_tuple
        result_str='%13s'%(exp_name[exp_name.rfind('\\')+1:])
        result_str+=', testing:  '+test_str
        result_str+=', training: '+train_str
        print 'run_num:%d %s'%(run_num,result_str)
        #write to result file
        dirname=result_filename[:result_filename.rfind('\\')]
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        origin_file=open(file_name)
        contents=origin_file.readlines()
        train_result_dict={}
        val_result_dict={}
        result_file=open(result_filename,'w')
        result_file.write(result_str+'\n\n')
        for oneline in contents[:10]:
            result_file.write(oneline[32:])
            if 'Start' in oneline:
                break
        #write to result file
        for idx in range(len(train_top1)):
            line='Epoch[%03d] %8s loss: %8.6f, acc: %6.3f; '%(idx+1,'training',train_loss[idx],train_top1[idx])
            line+='%8s loss: %05.3f, acc: %6.3f\n'%('testing',test_loss[idx],test_top1[idx])
            result_file.write(line)
        result_file.close()

def get_exps(snapshot_dir,network_name):
    txt_files=glob.glob('%s/%s/*.txt'%(snapshot_dir,network_name))
    exp_names={}
    for file_name in txt_files:
        key=file_name[:file_name.rfind('_')]
        depth=0
        if key not in exp_names.keys():
            try:
                depth=int(key[key.find('_d')+2:key.find('w')])
            except:
                depth+=1
            finally:
                exp_names[key]=depth
    import operator
    sorted_x = sorted(exp_names.items(), key=operator.itemgetter(1))
    return [name for name,depth in sorted_x]   

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--log-dir", type=str, default='./logfiles/cifar10',
                  help="directory of the log files, e.g. '--log-dir=./logfiles/cifar10'")
    parser.add_option("-f", "--force",
                  action="store_true", dest="override", default=False,
                  help="override existing files")
    args,_ = parser.parse_args()

    snapshot_dir=args.log_dir
    networks=['plain','side','fuse3','fuse6','resnet','half','cross']
    for network in networks:
        print 'network:'+network
        exp_names=get_exps(snapshot_dir,network)
        process_exps(snapshot_dir, exp_names,override=args.override)
        print '\n'
        
if __name__ == '__main__':
    main()