import glob
import numpy as np, os, sys
from scipy.io import loadmat,savemat
from scipy.signal import butter, lfilter,filtfilt,resample
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import torch
import torch.utils.data as Data
import numpy as np
import shutil
from itertools import cycle

#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
'''
(1)载入数据标签，
(2)滤波
（？）resample
(3)切分片段（
4）label 转换
（5）构建数据集 （
6）交叉验证
'''

def load_label(filename):
    '''
    (1)载入标签，
    get label from '*.hea' file
    :param filename: path of the hea(der) file
    :return:
    '''
    labels=[]
    with open(filename, 'r') as f:
        for l in f:
            if l.startswith('#Dx') or l.startswith('#DX'):
                tmp = l.split(': ')[1].split(',')
                for c in tmp:
                    labels.append(c.strip())
    return labels

def load_data(filename):
    '''
    (1)载入数据
    load data form '*.mat' file
    :param filename: path of mat file
    :return:
    '''
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    return data
def bandpass_filter(data, lowcut=1, highcut=45, signal_freq=500, filter_order=4,request_freq=250):
    """
    (2)滤波 +resample（？）
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y=filtfilt(b,a,data)
    #y = lfilter(b, a, data)
    if request_freq!=signal_freq:
        y=resample(y,y.shape[-1]*request_freq//signal_freq,axis=-1)
        #y=resample(y,len(y)*request_freq//signal_freq)
    return y

def data_slice(data,slice=6,overlap=4,freq=500):
    overlap_length=overlap*freq
    slice_length=slice*freq
    data_length=data.shape[-1]
    print(data_length)
    signal=data[:,:slice_length]
    all_signal=signal[None,:,:]
    cur_ind=slice_length
    while(cur_ind+(slice-overlap)*freq<=data_length):
        print(cur_ind)
        cur_signal=data[:,cur_ind-overlap*freq:cur_ind+(slice-overlap)*freq]
        cur_signal=cur_signal[None,:,:]
        cur_ind=cur_ind+(slice-overlap)*freq
        all_signal=np.concatenate([all_signal,cur_signal],axis=0)

    return all_signal
def slice_data(input_dir,output_dir,overlap=4):
    '''

    :param data:
    :param label:
    :return:
    '''
    search_value=input_dir+'*.mat'
    mat_files=glob.glob(search_value)
    for file in mat_files:
        print(file)
        record_name=(file.split('/')[-1]).split('.')[0]
        print(record_name)

        data=loadmat(file)['val']
        sliced_data=data_slice(data)
        for i,s_data in enumerate(sliced_data):
            mat_full_path=output_dir+record_name+'_%d'%(i)+'.mat'
            hea_full_path=mat_full_path.replace('mat','hea')
            src_hea_path=file.replace('mat','hea')
            print(mat_full_path,hea_full_path,src_hea_path)
            savemat(mat_full_path,{'val':s_data})
            shutil.copy(src_hea_path,hea_full_path)



def replace_equivalent_classes(classes, equivalent_classes):
    for j, x in enumerate(classes):
        for multiple_classes in equivalent_classes:
            if x in multiple_classes:
                classes[j] = multiple_classes[0] # Use the first class as the representative class.
    return classes

def labels2vector(classes,labels,equivalent_classes):
    '''

    transform smote label to vector labels
    :param target_labels: labels belongs to a record
    :return: vectorized label that only contain 1 and 0, means this record have one disease or not .
    '''
    vec_label=np.zeros(len(classes))
    labels=replace_equivalent_classes(labels,equivalent_classes)
    #print(len(total_labels))
    #total_labels=loadmat('./classes.mat')['classes']
    for i,label in enumerate(classes):
        if label in labels:
            vec_label[i]=1
    return vec_label

def DataLengthOperation(input_signal,fix_length=5000):
    '''
    to make sure all data have same length
    if current data is shorter than fixed length , then will padding it with 0
    if current data is longer than fixed length , then will cut signal to the fixed length
    the shape of input signal supposed to be (channel,length)
    :param fix_length:
    :return:
    '''
    assert len(input_signal.shape)==2 ,'The shape of input signal supposed to be (channel,length) in a numpy array '
    if input_signal.shape[1]>=fix_length:
        fix_signal=input_signal[:,:fix_length]
    else:
        fix_signal=np.pad(input_signal,((0,0),(0,fix_length-input_signal.shape[1])),'constant',constant_values=0)

    return fix_signal


def DataProcessPipeline(filename,fix_length=2500+1596):
    #classes=['270492004','164889003','164909002','284470004','59118001','426783006','429622005','164931005','164884008']
    #classes=['270492004','164889003','164909002','284470004','59118001','426783006']
    #file_full_path=input_directory+filename
    classes=['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002', '39732003', '164909002', '251146004', '698252002', '10370003', '284470004', '427172004', '164947007', '111975006', '164917005', '47665007', '427393009', '426177001', '426783006', '427084000', '164934002', '59931005']

    equivalent_classes=[[ '59118001','713427006',], ['284470004', '63593006'], ['427172004', '17338001']]

    file_full_path=filename
    label_full_path=filename.replace('mat','hea')

    data  = load_data(file_full_path)


    label = load_label(label_full_path)
    data = bandpass_filter(data)  # if want change filter configration,go to bandpass_filter and make changes
    data = DataLengthOperation(data, fix_length=fix_length)
    vec_label = labels2vector(classes,label,equivalent_classes)#todo 获取classes
    return data ,vec_label
class myDateset(torch.utils.data.Dataset):
    def __init__(self,root,domain=0,mean=0,std=1,Max=0,Min=0,norm=True):
        self.root=root+'*.mat'
        self.domain=domain
        self.mean=mean
        self.std=std
        self.Max=Max
        self.Min=Min
        self.files=np.array(glob.glob(self.root))
        self.norm=norm
    def __getitem__(self, index):
        filename=self.files[index]
        #print(filename)
        data,label=DataProcessPipeline(filename)
        data=torch.from_numpy(data.astype('float32'))
        label=label.astype('float32')
        if self.norm:
            #print(data.shape)
            #print(data.min(),data.max())
            #data=(data-self.mean)/self.std
            data=(data-data.min())/(data.max()-data.min())
            data=data.float()
        domain=torch.as_tensor(self.domain,dtype=torch.long)

        return data,label#,domain
    def __len__(self):
        return len(self.files)

def cal_cls_fusion_factor(source_iters,num_classes=24):
    sample_count=torch.zeros((1,num_classes))
    for iter in source_iters:
        cur_sample_count=torch.zeros((1,num_classes))
        for _,y in  iter:
            cur_sample_count+=torch.sum(y,dim=0)
        sample_count=torch.cat([sample_count,cur_sample_count],dim=0)

    sample_count=sample_count[1:][:]
    sample_count/=(torch.sum(sample_count,dim=0)+1)
    print(sample_count)
    weight=sample_count.float()
    # weight=torch.softmax(sample_count.float(),dim=0)
    # print(weight)
    return weight

def cal_cls_fusion_factor_ratio(source_iters,num_classes=24):
    sample_count=torch.zeros((1,num_classes))
    total_count=torch.zeros(3)
    for i,iter in enumerate(source_iters):
        cur_sample_count=torch.zeros((1,num_classes))
        for _,y in  iter:
            cur_sample_count+=torch.sum(y,dim=0)
            total_count[i]+=y.shape[0]
        sample_count=torch.cat([sample_count,cur_sample_count],dim=0)

    sample_count=sample_count[1:][:]
    #sum=torch.sum(sample_count,dim=1)
    l=total_count.reshape(sample_count.shape[0],-1).expand(-1,24)
    print('l.shape:',l.shape,l)
    sample_count/=l
    #sum=torch.sum(sample_count,dim=1)+1e-9
    sample_count/=(torch.sum(sample_count,dim=0)+1e-9)
    print(sample_count)
    weight=sample_count.float()
    # weight=torch.softmax(sample_count.float(),dim=0)
    # print(weight)
    return weight



def CreateDataset(input_directories,domains=[0,1,2,3,4,5,6]):
    for i,input_directory in enumerate(input_directories):
        print(input_directory,os.path.exists(input_directory))
        if i == 0:
            dataset=myDateset(input_directory,domain=domains[i])
            data_mean,data_var,data_max,data_min=DataStatistic(dataset)
            dataset=myDateset(input_directory,data_mean,data_var,data_max,data_min,norm=True)
            continue
        cur_set=myDateset(input_directory,domain=domains[i])
        data_mean, data_std, data_max, data_min = DataStatistic(cur_set)
        cur_set = myDateset(input_directory,i,data_mean, data_std,data_max,data_min, norm=True)
        dataset=Data.ConcatDataset([dataset,cur_set])

    return dataset
def CreateDatasetIters(input_directory,batch_size=64,shuffle=True,num_workers=8,norm=False,requried_Val_iter=False,val_ratio=0.2):
    '''创建用于WMSUNN的迭代器'''
    #print(input_directory)
    dataset=myDateset(input_directory)
    if norm:
        data_mean, data_var, data_max, data_min = DataStatistic(dataset)
        dataset = myDateset(input_directory, data_mean, data_var, data_max, data_min, norm=True)

    if not requried_Val_iter:
        data_iter=Data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=True)
        #cycle_iter=cycle(data_iter)
        return data_iter
    else:
        trainset,valset=Data.random_split(dataset,[len(dataset)-int(val_ratio*len(dataset)),int(val_ratio*len(dataset))])
        train_iter=Data.DataLoader(trainset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=True)
        val_iter=Data.DataLoader(valset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=True)
        return train_iter,val_iter

def CreateDataset_DANN(input_directories,domains=[0,1,2,3,4,5,6]):
    for i,input_directory in enumerate(input_directories):
        print(input_directory,os.path.exists(input_directory))
        if i == 0:
            dataset=myDateset(input_directory,domain=domains[i])
            data_mean,data_var,data_max,data_min=DataStatistic_DANN(dataset)
            dataset=myDateset(input_directory,data_mean,data_var,data_max,data_min,norm=True)
            continue
        cur_set=myDateset(input_directory,domain=domains[i])
        data_mean, data_std, data_max, data_min = DataStatistic_DANN(cur_set)
        cur_set = myDateset(input_directory,i,data_mean, data_std,data_max,data_min, norm=True)
        dataset=Data.ConcatDataset([dataset,cur_set])

    return dataset

def CreateKFoldDataset(dataset,K,batchsize=32,num_worker=8,shuffle=True):
    length = len(dataset)

    fold_length = length // K
    fold_list = []
    for i in range(K - 1):
        fold_list.append(fold_length)
    fold_list.append(length - (K - 1) * fold_length)
    splited_datasets = Data.random_split(dataset=dataset, lengths=fold_list)
    KFoldsets = []
    for i, dataset in enumerate(splited_datasets):
        a = [x for x in range(K)]
        a.pop(i)

        val_set = dataset
        train_set = Data.ConcatDataset([splited_datasets[j] for j in a])


        train_loader = Data.DataLoader(train_set, batch_size=batchsize, num_workers=num_worker, shuffle=shuffle)
        val_loader = Data.DataLoader(val_set, batch_size=batchsize, num_workers=num_worker, shuffle=shuffle)
        Foldset = (train_loader, val_loader)
        KFoldsets.append(Foldset)

    return KFoldsets
def DataStatistic(dataset):
    print(len(dataset))
    data_iter=Data.DataLoader(dataset,batch_size=256,num_workers=16,shuffle=False)
    for i,(x,_) in enumerate(data_iter):
        print(i,'/',len(data_iter))
        if i==0:
            all_data=x
            continue
        all_data=torch.cat([all_data,x],dim=0)

    data_mean,data_std,data_max,data_min=all_data.mean(),all_data.std(),all_data.max(),all_data.min()
    print(data_mean,data_std,data_max,data_min)
    del all_data
    return data_mean,data_std,data_max,data_min
def DataStatistic_DANN(dataset):
    print(len(dataset))
    data_iter=Data.DataLoader(dataset,batch_size=256,num_workers=16,shuffle=False)
    for i,(x,_,_) in enumerate(data_iter):
        #print(x)
        print(i,'/',len(data_iter))
        if i==0:
            all_data=x
            continue
        all_data=torch.cat([all_data,x],dim=0)
    print(all_data.shape)

    data_mean,data_std,data_max,data_min=all_data.mean(),all_data.std(),all_data.max(),all_data.min()
    print(data_mean,data_std,data_max,data_min)
    return data_mean,data_std,data_max,data_min

if __name__ == '__main__':
    test_directory = [
        './dataset/CPSC/',]
    from itertools import cycle
    ds=CreateDataset(test_directory)
    testloader=Data.DataLoader(ds,batch_size=256,num_workers=8)
    t=cycle(testloader)
    for i in range(500):
        print(i,(t.__next__())[0].shape)
        #print()




    #slice_data('./dataset/Training_CPSC/','./slicedCSPC/')
    # a=np.random.random([12,1000])
    # fix_a=DataLengthOperation(a)
    # print(fix_a.shape)
    # splited_a=data_slice(a)
    # print(a.shape,splited_a.shape)
    # a=CreateDataset(['./dataset/PhysioNetChallenge2020_PTB-XL/WFDB/'])
    # DataStatistic(a)



