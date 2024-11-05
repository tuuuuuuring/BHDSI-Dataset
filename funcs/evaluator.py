import numpy as np

def LOSS_OR(pre, label):
    pre_numpy = pre.cpu().detach().numpy()
    label_numpy = label.cpu().detach().numpy()
    a = np.maximum(pre_numpy / label_numpy, label_numpy / pre_numpy)

    a[label_numpy == 0] = 0
    a[np.logical_and(a < 1.25, a > 0)] = 1
    a[a >= 1.25] = 0
    count_t = np.sum(a)
    label_numpy[label_numpy > 0] = 1
    result = count_t / np.sum(label_numpy)
    return result

def LOSS_OR_2(pre, label):
    pre_numpy=pre.cpu().detach().numpy()
    label_numpy=label.cpu().detach().numpy()
    a=np.maximum(pre_numpy/label_numpy,label_numpy/pre_numpy)
    a[label_numpy==0]=0
    a[np.logical_and(a<1.25*1.25,a>0)]=1
    a[a>=1.25*1.25]=0
    count_t=np.sum(a)
    label_numpy[label_numpy>0]=1
    result=count_t/np.sum(label_numpy)
    return result

def LOSS_OR_3(pre, label):
    pre_numpy=pre.cpu().detach().numpy()
    label_numpy=label.cpu().detach().numpy()
    a=np.maximum(pre_numpy/label_numpy,label_numpy/pre_numpy)
    a[label_numpy==0]=0
    a[np.logical_and(a<1.25*1.25*1.25,a>0)]=1
    a[a>=1.25*1.25*1.25]=0

    count_t=np.sum(a)
    label_numpy[label_numpy>0]=1
    result=count_t/np.sum(label_numpy)
    return result

def REL(pre, label):
    pre_numpy = pre.cpu().detach().numpy()
    label_numpy = label.cpu().detach().numpy()
    deta = np.abs(pre_numpy-label_numpy)/label_numpy

    deta[label_numpy==0]=0
    deta_sum=np.sum(deta)
    label_numpy[label_numpy>0]=1
    result=deta_sum/np.sum(label_numpy)
    return result