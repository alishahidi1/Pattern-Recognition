#Importing required libraries
import numpy as np
import matplotlib.pyplot as plt

#Importing Tic-Toc-Toe dataset
ds=np.loadtxt('tic-tac-toe.data', delimiter=',', dtype='str')

#Defining the array of attributes
fs=np.arange(ds[:,:-1].shape[1])

#Function for Calculating the entropy
def h(t): 
    y=0
    #v is the array of different values of t, c is the number of each value
    v,c=np.unique(t, return_counts=True)
    N=np.sum(c)
    for i in range(len(v)):
        y=y+(-c[i]/N*np.log2(c[i]/N))
    return y

#-----------------------------------------------------#


#Function for Calculating the Information Gain for categorical dataset(Tic-Toc-Toe)
#d is the dataset and atr is the desired attribute
def IG(d, atr):
    
    #H is the total entropy
    H=h(d[:,-1])
    
    #Values of the specified attribute and their coresponding counts
    v,c=np.unique(d[:,atr], return_counts=True)
    
    ph=0
    N=np.sum(c)
    
    for i in range(len(v)):
        ph+=c[i]/N*h(d[np.where(d[:,atr]==v[i])[0]][:,-1])
    
    return H-ph

#-------------------------------------------------------#
#Function for calculating Gain Ratio of categorical dataset(Tic-Toc-Toe)
def GR(d, atr):
    v,c=np.unique(d[:,atr], return_counts=True)
    si=0.0
    N=np.sum(c)
    G=0
    for i in range(len(v)):
        si-=c[i]/N*np.log2(c[i]/N)
        G+=c[i]/N*h(d[np.where(d[:,atr]==v[i])[0]][:,-1])
    if si==0.0:
        si=1e-3
    H=h(d[:,-1])
    return (H-G)/si
#--------------------------------------------------------#

# Defining a class type variable to be used for ....
#constructing decision tree of categorical dataset(Tic-Toc-Toe)
class DT_cat:
    def __init__(self, atri=-1, va=[], maxtgt=None, trg=[], cl=[]):
        #atri=attribute or node, va=values, 
        #maxtgt=most common class label
        #trg=list of class labels(with respect to node values)
        #cl=children classes
        self.atri=atri
        self.va=va
        self.trg=trg
        self.cl=cl
        self.maxtgt=maxtgt

# Function for constructing ID3 Decision Tree for 
# categorical dataset(Tic-Toc-Toe)
def DT_ID3_class(d, d_org, atr, p_max=None, base=0):
    #d=dataset, d_org=original dataset
    #atr=array of attributes, p_max=most common label
    #base=0 then Information Gain is used(default)
    #base=1 then Gain Ratio is used
    l=np.unique(d[:,-1], return_counts=True) 
    #l[0]=existing labels, l[1]= the counts of each label
    if len(l[0])==1:
        #if all of the labels where equal makes the leaf
        return l[0][0]
    elif len(d)==0:
        #if d is empty returns the most common label
        l_org=np.unique(d_org[:,-1], return_counts=True)
        i_max=np.argmax(l_org[1])
        return l_org[0][i_max]
    elif len(atr)==0:
        #if no attribute is remained return the most common label
        #of the currenct node
        return p_max
    
    else:
        #first, the most common class label is obtained
        if len(np.unique(l[1]))==1:
            #if counts of different class labels are equal,
            #most common label is selected randomly
            p_max=l[0][np.random.randint(len(l[0]))]
        else:
            i_max=np.argmax(l[1])
            p_max=l[0][i_max]
        
        v=np.array([])
        
        #calculating Information Gain or Gain Ratio of each
        #attribute(depending on the question)
        for i in atr:
            if base==0:
                v=np.append(v, IG(d, i))
            else:
                v=np.append(v, GR(d, i))
        #selecting the node with maximum IG or GR
        atr_i=np.argmax(v)
        atr_max=atr[atr_i]
        
        #making a class for this node
        y=DT_cat(va=[], trg=[], cl=[])
        y.atri=atr_max
        y.maxtgt=p_max
        
        #Obtaining different values of the node
        vals=np.unique(d[:,atr_max])
        q=0
        
        #deleting the current attribute from the attribute array
        atr=np.delete(atr,atr_i)
        for i in vals:
            y.va.append(i)
            y.trg.append([])
            y.cl.append([])
            
            # making a new dataset for each value of the node
            d_new=d[np.where(d[:,atr_max]==i)[0],:]
            
            #making a new branch for the tree
            y_new=DT_ID3_class(d_new, d_org, atr, p_max, base)
            
            #checking if leaf is made
            if type(y_new)!=DT_cat:
                y.trg[q]=y_new
                #y.va.append(i)
            else:
                y.cl[q]=y_new
            q+=1
        #returning the decision tree
        return(y)
#----------------------------------------------------#

#Importing Wine dataset
dc=np.loadtxt('wine.data', delimiter=',', dtype='float')
dc=np.insert(dc,len(dc[0,:]),dc[:,0],axis=1)
dc=np.delete(dc,0,axis=1)

#Defining the array of attributes
fc=np.arange(dc[:,:-1].shape[1])

#Function for Calculating the Information Gain #for numerical dataset(Wine)
#d=dataset, atr=desired attribute, trs=threshold    
def IG_c(d, atr, trs):
    
    #H is the total entropy
    H=h(d[:,-1])
    
    #d1=array of indexes for values less than threshold
    #d2=array of indexes for values more than or euqal #to threshold
    d1=np.where(d[:,atr]<trs)[0]
    d2=np.where(d[:,atr]>=trs)[0]
    dn=[d1,d2]
    c=[len(d1),len(d2)]
    ph=0
    N=sum(c)
    
    for i in range(2):
        ph=ph+c[i]/N*h(d[dn[i],-1])
    #ph is the conditional entropy
    return H-ph
#----------------------------------------------#
#Function for calculating Gain Ratio of numerical dataset(Wine)
def GR_c(d, atr, trs):
    #d1=array of indexes for values less than threshold
    #d2=array of indexes for values more than or euqal #to threshold
    d1=np.where(d[:,atr]<trs)[0]
    d2=np.where(d[:,atr]>=trs)[0]
    dn=[d1,d2]
    c=[len(d1),len(d2)]
    si=0
    G=0
    N=sum(c)
    for i in range(2):
        if c[i]!=0:
            si-=c[i]/N*np.log2(c[i]/N)
            G+=c[i]/N*h(d[dn[i],-1])
    if si==0:
        si=1e-3
    #si=Split Information
    H=h(d[:,-1])
    return (H-G)/si
#----------------------------------------------#

#Defining function for obtaining the threshold...
#with maximum Information Gain
def IG_c_max(d,atr):
    #arg_i=index of minimum value
    arg_i=np.argmin(d[:,atr])
    trs=d[arg_i,atr]
    IG=IG_c(d,atr,trs)
    tg=d[arg_i,-1]
    val_prev=d[arg_i,atr]
    #sorting the values of the attribute
    #calculating the information gain where...
    #class label changes
    for i in np.argsort(d[:,atr]):
        tg_new=d[i,-1]
        if tg_new!=tg:
            tg=tg_new
            tr_new=(d[i,atr]+val_prev)/2
            IG_new=IG_c(d,atr,tr_new)
            #Obtaining the threshold with maximum value
            if IG_new>IG:
                IG=IG_new
                trs=tr_new
        val_prev=d[i,atr]
    #IG=maximum value of Information Gain
    #trs=Threshold with maximum IG
    return IG,trs

#----------------------------------------------#
#Defining function for obtaining the threshold...
#with maximum Gain Ratio
def GR_c_max(d, atr):
    #arg_i=index of minimum value
    arg_i=np.argmin(d[:,atr])
    trs=d[arg_i,atr]
    GR=GR_c(d,atr,trs)
    tg=d[arg_i,-1]
    val_prev=d[arg_i,atr]
    #sorting the values of the attribute
    #calculating the gain ratio where...
    #class label changes
    for i in np.argsort(d[:,atr]):
        tg_new=d[i,-1]
        if d[i,-1]!=tg:
            tg=tg_new
            tr_new=(d[i,atr]+val_prev)/2
            GR_new=GR_c(d,atr,tr_new)
            #Obtaining the threshold with maximum value
            if GR_new>GR:
                GR=GR_new
                trs=tr_new
        val_prev=d[i,atr]
    #GR=maximum value of Gain Ratio
    #trs=Threshold with maximum GR
    return GR,trs

#----------------------------------------------#

# Defining a class type variable to be used for ....
#constructing decision tree of numerical dataset(Wine)
class DT_c:
    def __init__(self, atri=-1, tres=None, trg=[], cl=[]):
        #atri=attribute or node, 
        #trg=list of class labels:
            #trg[0]=value<tres
            #trg[1]=value>tres
        #cl=children classes
        self.atri=atri
        self.trg=trg
        self.cl=cl
        self.tres=tres
        

# Function for constructing ID3 Decision Tree for.. 
# numerical dataset(Wine)        
def DT_ID3_class_c(d, d_org, atr, p_max=None, base=0):
    #d=dataset, d_org=original dataset
    #atr=array of attributes, p_max=most common label
    #base=0 then Information Gain is used(default)
    #base=1 then Gain Ratio is used
    l=np.unique(d[:,-1], return_counts=True)
    #l[0]=existing labels, l[1]= the counts of each label
    if len(l[0])==1:
        #if all of the labels where equal makes the leaf
        return l[0][0]
    elif len(d)==0:
        #if d is empty returns the most common label
        l_org=np.unique(d_org[:,-1], return_counts=True)
        i_max=np.argmax(l_org[1])
        return l_org[0][i_max]
    elif len(atr)==0:
        #if no attribute is remained return the most common label
        #of the currenct node
        return p_max
    
    else:
        #first, the most common class label is obtained
        if  len(np.unique(l[1])==1): #len(l[0])==2 and l[1][0]==l[1][1]:
            #if counts of different class labels are equal,
            #most common label is selected randomly
            p_max=l[0][np.random.randint(len(l[0]))]
        else:
            i_max=np.argmax(l[1])
            p_max=l[0][i_max]
        
        v=np.array([])
        tr=np.array([])
        #print(d)
        
        #calculating Information Gain or Gain Ratio of each
        #attribute(depending on the question)
        for i in atr:
            if base==0:
                IG=IG_c_max(d,i)
            else:
                IG=GR_c_max(d,i)
            v=np.append(v, IG[0])
            tr=np.append(tr, IG[1])
        #selecting the node with maximum IG or GR
        atr_i=np.argmax(v)
        atr_max=atr[atr_i]
        #tr_max=threshold of the selected node
        tr_max=tr[atr_i]
        
        #making a class for this node
        y=DT_c(atri=atr_max, tres=tr_max, trg=[[],[]], cl=[[],[]])
        q=0
        
        #deleting the current attribute from the attribute array
        atr=np.delete(atr,atr_i)
        
        #making 2 branches:
            #d1=data samples with node values less than the threshold
            #d2=data samples with node values more than or...
            #equal to the threshold
        d1=d[np.where(d[:,atr_max]<tr_max)[0],:]
        d2=d[np.where(d[:,atr_max]>=tr_max)[0],:]
        

        dn=[d1,d2]
        for i in range(2):
            #d_new=dataset of the new branch
            d_new=dn[i]
            #y_new=new branch
            y_new=DT_ID3_class_c(d_new, d_org, atr, p_max, base)
            
            #check if the leaf is made
            if type(y_new)!=DT_c:
                y.trg[q]=y_new
            else:
                #make the children class of tree
                y.cl[q]=y_new
            q+=1
        
        #returning the decision tree
        return y
#---------------------------------------------#

#Defining a function to predict class labels...
#using sample feature values & the constructed..
#decision tree as inputs

#pr_class: predicts the class label for categorical..
#datasets(Tic-Toc-Toe)
def pr_class(v,tr):
    #v=array of feature values of the sample
    #tr=decision tree class
    atr=tr.atri
    val=v[atr]
    try:
        #check if the value exist in the decision tree
        ind=tr.va.index(val)
    except:
        #returns maximum class label in case the value
        #didn't exist
        return tr.maxtgt
    
    if isinstance(tr.trg[ind],str):#tr.trg[ind]!=[]:
        #checks if the leaf exists
        return tr.trg[ind]
    else:
        #recursively calls the function for the children...
        #decision tree class
        return pr_class(v,tr.cl[ind])
    
#--------------------------------------------#    
#pr_class_c: predicts the class label for numerical..
#datasets(Wine)
def pr_class_c(v,tr):
    #v=array of feature values of the sample
    #tr=decision tree class
    atr=tr.atri
    trs=tr.tres
    val=v[atr]
    if val<trs:
        j=0
    else:
        j=1
    if isinstance(tr.trg[j],float):
        #checks if the leaf exists
        return tr.trg[j]
    else:
        #recursively calls the function for the children...
        #decision tree class
        return pr_class_c(v,tr.cl[j])
    
#--------------------------------------------#

#Defining a function for splitting data samples to..
#a training set & a test set
def split(d,k,i):
    #d=dataset, k=number of folds, i=index of test fold
    n=len(d)//k
    if i<k-1:
        d_test=d[i*n:i*n+n]
        d_train=np.delete(d, np.s_[i*n:i*n+n],axis=0)
    else:
        d_test=d[i*n:]
        d_train=d[:i*n]#np.delete(d, np.s_[i*n:],axis=0)
    return d_train, d_test

#--------------------------------------------#

#Defining a function for calculating the accuarcy..
#and the confusion matrix for categorical dataset#(Tic-Toc-Toe)
def Accuracy(train,test, bs=0):
    #train=training set of samples
    #test=test set of samples
    #bs=0 then Information Gain is used
    #bs=1 then Gain Ratio is used
    #this function has 2 outputs:
        #1:Percentage of Accuracy
        #2:The Confusion Matrix
    
    #f=array of attributes
    f=np.arange(train[:,:-1].shape[1])
    
    #constructing the decision tree
    tree=DT_ID3_class(train,train,f,base=bs)
    N=len(test)
    v,c=np.unique(train[:,-1], return_counts=True)
    m=len(v)
    dic={}
    q=0
    
    #constructing the confusion matrix
    for i in range(m):
        dic[v[i]]=q
        q+=1
        
    cm=np.zeros((m,m), dtype=int)
    
    #predicted labels are compares with the actual labels..
    #then the accuracy and the confusion matrix are obtained
    for i in range(N):
        tr=test[i,-1]
        pr=pr_class(test[i,:-1],tree)
        cm[dic[tr],dic[pr]]+=1
    return np.trace(cm)/N*100,cm
#--------------------------------------------#

#Defining a function for calculating the accuarcy..
#and the confusion matrix for numerical dataset#(Wine)
def Accuracy_c(train,test, bs=0):
    #train=training set of samples
    #test=test set of samples
    #bs=0 then Information Gain is used
    #bs=1 then Gain Ratio is used
    #this function has 2 outputs:
        #1:Percentage of Accuracy
        #2:The Confusion Matrix
    
    #f=array of attributes
    f=np.arange(train[:,:-1].shape[1])
    
    #constructing the decision tree
    tree=DT_ID3_class_c(train,train,f, base=bs)
    
    N=len(test)
    v,c=np.unique(train[:,-1], return_counts=True)
    m=len(v)
    dic={}
    q=0
    
    #constructing the confusion matrix
    cm=np.zeros((m,m), dtype=int)
    for i in range(m):
        dic[v[i]]=q
        q+=1
    
    #predicted labels are compares with the actual labels..
    #then the accuracy and the confusion matrix are obtained
    for i in range(N):
        tr=test[i,-1]
        pr=pr_class_c(test[i,:-1],tree)
        cm[dic[tr],dic[pr]]+=1
    return np.trace(cm)/N*100,cm
#--------------------------------------------#

#Defining a function for doing K-fold-t-times cross validation..
#for categorical dataset(Tic-Toc-Toe)
def k_fold_t_times(d,k=10,t=10,trprc=0,tsprc=0,ntype=0,bss=0):
    #d=dataset, k=number of folds, t= number of times
    #trprc=percentage of noise for training set
    #tspric=percentage of noise for test set
    #bss: 0:Information Gain, 1:Gain Ratio
    #ntype=noise type:
        #0:attribute noise
        #1:contradictive examples noise
        #2: Misclassification noise
        
        
    acc=np.zeros(k*t)
    N=len(d)
    n=k*t
    best_acc=0.0
    q=0
    
    #loop for times
    for i in range(t):
        
        #data set is shuffled before doing k-fold cross validation..
        #each time
        
        d_new=d[np.random.permutation(N)]
        
        #loop for folds
        for j in range(k):
            #splitting data samples to training & test sets
            d_split=split(d_new,k,j)
            
            #applying the type and percentage of desired noise
            if ntype==0:
                d_train=noise_atr(d_split[0],trprc)
                d_test=noise_atr(d_split[1],tsprc)
            elif ntype==1:
                d_train=noise_cont(d_split[0],trprc)
                d_test=noise_cont(d_split[1],tsprc)
            else:
                d_train=noise_mis(d_split[0],trprc)
                d_test=noise_mis(d_split[1],tsprc)
            
            #Calculating the accuracy
            acrs=Accuracy(d_train,d_test, bs=bss)
            
            #acc=array of all accuracies
            acc[q]=acrs[0]
            
            #Calculating the confusion matrix with respect to..
            #... the maximum accuracy
            if acc[q]>best_acc: 
                best_acc=acc[q]
                cm=acrs[1]
            q+=1
    
    #this function has 4 outputs:
        #1:average accuracy(%)
        #2:variance of the accuracy
        #3: Confusion Matrix with respect to best accuracy
        #4: the best achieved accuracy
    return np.mean(acc),variance(acc)*1e-4,cm,best_acc

#--------------------------------------------#

#Defining a function for doing K-fold-t-times cross validation..
#for numerical dataset(Wine)
def k_fold_t_times_c(d,k=10,t=10,trprc=0,tsprc=0,ntype=0, bss=0):
    #d=dataset, k=number of folds, t= number of times
    #trprc=percentage of noise for training set
    #tspric=percentage of noise for test set
    #bss: 0:Information Gain, 1:Gain Ratio
    #ntype=noise type:
        #0:attribute noise
        #1:contradictive examples noise
        #2: Misclassification noise
        
    N=len(d)
    n=k*t
    acc=np.zeros(k*t)
    q=0
    best_acc=0.0
    
    #loop for times
    for i in range(t):
        
        #data set is shuffled before doing k-fold cross validation..
        #each time
        
        d_new=d[np.random.permutation(N)]
        
        #loop for folds
        for j in range(k):
            
            #splitting data samples to training & test sets
            d_split=split(d_new,k,j)
            
            #applying the type and percentage of desired noise
            if ntype==0:
                d_train=noise_atr_c(d_split[0],d,trprc)
                d_test=noise_atr_c(d_split[1],d,tsprc)
            elif ntype==1:
                d_train=noise_cont(d_split[0],trprc)
                d_test=noise_cont(d_split[1],tsprc)
            else:
                d_train=noise_mis(d_split[0],trprc)
                d_test=noise_mis(d_split[1],tsprc)
            
            #Calculating the accuracy    
            acrs=Accuracy_c(d_train,d_test, bs=bss)
            
            #acc=array of all accuracies
            acc[q]=acrs[0]
            
            #Calculating the confusion matrix with respect to..
            #... the maximum accuracy
            if acc[q]>best_acc:
                best_acc=acc[q]
                cm=acrs[1]
            q+=1
    #this function has 4 outputs:
        #1:average accuracy(%)
        #2:variance of the accuracy
        #3: Confusion Matrix with respect to best accuracy
        #4: the best achieved accuracy
    return np.mean(acc),variance(acc)*1e-4,cm,best_acc

#-------------------------------------------------#

def variance(x):
    return np.mean(np.power((x-np.mean(x)),2))

#-------------------------------------------------#

#noise_atr: function for applying attribute noises...
#to categorical dataset(Tic-Toc-Toe)

def noise_atr(d,prc):
    #d=data samples
    #prc=noise percentage
    
    dd=d.copy()
    
    if prc==0: #In case of clean dataset(no noise)
        return dd
    else:
        hh=len(dd)
        
        #s=number of noisy samples
        s=int(prc*hh)+1
        
        #ind=indices of randomly selected samples..
        #..for applying noise
        ind=np.random.choice(hh,s, replace=False)
        ff=np.arange(dd[:,:-1].shape[1])
        v=np.array(['x','o','b'])
        
        #loop for applying noise to each attribute
        for i in range(len(ff)):
            #randomly selecting a value of the attribute...
            #and assigning it to the sample
            dd[ind,i]=v[np.random.randint(len(v),size=s)]
        return dd

#-------------------------------------------------#

#noise_atr: function for applying attribute noises...
#to numerical dataset(Wine)
def noise_atr_c(d,d_org,prc):
    #d=data samples
    #prc=noise percentage
    dd=d.copy()
    if prc==0: #In case of clean dataset(no noise)
        return dd
    else:
        hh=len(dd)
        
        #s=number of noisy samples
        s=int(prc*hh)+1
        
        #ind=indices of randomly selected samples..
        #..for applying noise
        ind=np.random.choice(hh,s, replace=False)
        ff=np.arange(dd[:,:-1].shape[1])
        
        #loop for applying noise to each attribute
        for i in range(len(ff)):
            
            #randomly selecting numbers from zero mean....
            #.. normal distribution
            noi=np.random.normal(loc=0, scale=1,size=s)
            
            #adding noise to the attribute values of selected..
            #..samples
            dd[ind,i]+=noi*(max(d_org[:,i])-min(d_org[:,i]))/2
        return dd

#-------------------------------------------------#

#noise_cont: a function for applying Contradictive..
#..examples noise
def noise_cont(d, prc):
    
    #d=data samples
    #prc=noise percentage
    dd=d.copy()
    if prc==0: #In case of clean dataset(no noise)
        return dd
    else:
        hh=len(dd)
        
        #s=number of noisy samples
        s=int(prc*hh)
        
        #ind=indices of randomly selected samples..
        #..for applying class label noise
        ind=np.random.choice(hh,s, replace=False)
        
        #v=array of class labels
        v=np.unique(dd[:,-1])
        
        #loop for changing class labels of the ...
        #.. the selected samples
        for i in ind:
            #vi=array of class labels without the actual class label
            vi=v[np.argwhere(v!=dd[i,-1])]
            
            #randomly selecting a different class label
            dd[i,-1]=vi[np.random.randint(len(vi))][0]
        
        #adding samples with changed class labels to the...
        #..original dataset
        dd=np.append(d,dd[ind,:],axis=0)
        return dd

#-------------------------------------------------#

#noise_mis: a function for applying Misclassifications..
#.. noise
def noise_mis(d,prc):
    
    #d=data samples
    #prc=noise percentage
    dd=d.copy()
    if prc==0: #In case of clean dataset(no noise)
        return dd
    else:
        hh=len(dd)
        
        #s=number of noisy samples
        s=int(prc*hh)
        ind=np.random.choice(hh,s, replace=False)
        
        #v=array of class labels
        v=np.unique(dd[:,-1])

        #loop for changing class labels of the ...
        #.. the selected samples
        for i in ind:
            
            #vi=array of class labels without the actual class label
            vi=v[np.argwhere(v!=dd[i,-1])]
            
            #randomly selecting a different class label
            dd[i,-1]=vi[np.random.randint(len(vi))][0]
        return dd
#-------------------------------------------------#

#Question3-Part(A)-Attribute Noise
#In this section, attribute noises are applied with...
#.. different percentages

#Tic-Toc-Toe Dataset:

noises=[0.05,0.1,0.15]

#acc1=list of average accuracies
acc1=[[],[],[],[]]

#var1=list of variances
var1=[[],[],[],[]]

#loop for runing 10-times-10-fold cross validation...
#..for different percentage and types of attribute noise
for i in noises:    
    
    #DvsC
    y0=k_fold_t_times(ds,trprc=i,tsprc=0,ntype=0)
    
    #DvsD
    y1=k_fold_t_times(ds,trprc=i,tsprc=i,ntype=0)
    
    #CvsD
    y2=k_fold_t_times(ds,trprc=0,tsprc=i,ntype=0)
    acc1[0].append(y0[0])
    acc1[1].append(y1[0])
    acc1[2].append(y2[0])
    var1[0].append(y0[1])
    var1[1].append(y1[1])
    var1[2].append(y2[1])

#CvsC
y3=k_fold_t_times(ds,trprc=0,tsprc=0,ntype=0)
acc1[3].append(y3[0])
var1[3].append(y3[1])

#Plotting the figure of Curves
plt.figure(figsize=(8,6))
for i in range(len(acc1)-1): plt.plot(np.multiply(noises,100),acc1[i],'*-', markersize=9)
plt.plot(0,acc1[3],'*', markersize=12)
plt.legend(('DvsC','DvsD','CvsD','CvsC'), fontsize=11, labelspacing=1, loc='lower left')
plt.xlabel('Noise(%)', fontsize=12)
plt.ylabel('Accuracy(%)', fontsize=12)
plt.title('Tic Tac Toe Dataset with Attribute Noise')
plt.xticks(np.arange(0,16,5),fontsize=12)
plt.yticks(np.arange(70,91,5),fontsize=12)
plt.grid()
plt.show()

#Wine Dataset:

noises=[0.05,0.1,0.15]

#acc2=list of average accuracies
acc2=[[],[],[],[]]

#var2=list of variances
var2=[[],[],[],[]]

#loop for runing 10-times-10-fold cross validation...
#..for different percentage and types of attribute noise
for i in noises:    
    
    #DvsC
    y0=k_fold_t_times_c(dc,trprc=i,tsprc=0,ntype=0)
    
    #DvsD
    y1=k_fold_t_times_c(dc,trprc=i,tsprc=i,ntype=0)
    
    #CvsD
    y2=k_fold_t_times_c(dc,trprc=0,tsprc=i,ntype=0)
    acc2[0].append(y0[0])
    acc2[1].append(y1[0])
    acc2[2].append(y2[0])
    var2[0].append(y0[1])
    var2[1].append(y1[1])
    var2[2].append(y2[1])

#CvsC
y3=k_fold_t_times_c(dc,trprc=0,tsprc=0,ntype=0)
acc2[3].append(y3[0])
var2[3].append(y3[1])

#Plotting the figure of Curves
plt.figure(figsize=(8,6))
for i in range(len(acc2)-1): plt.plot(np.multiply(noises,100),acc2[i],'*-', markersize=9)
plt.plot(0,acc2[3],'*', markersize=12)
plt.legend(('DvsC','DvsD','CvsD','CvsC'), fontsize=11, labelspacing=1, loc='lower left')
plt.xlabel('Noise(%)', fontsize=12)
plt.ylabel('Accuracy(%)', fontsize=12)
plt.title('Wine Dataset with Attribute Noise')
plt.xticks(np.arange(0,16,5), fontsize=12)
plt.yticks(np.arange(80,101,5), fontsize=12)
plt.grid()
plt.show()
#-------------------------------------------------#

#Question3-Part(B)-Class-label Noise

#In this section, class-label noises are applied with...
#.. different percentages

#Tic-Toc-Toe Dataset:
noises=[0.05,0.1,0.15]

#acc3=list of average accuracies
acc3=[[],[],[]]

#var3=list of variances
var3=[[],[],[]]

q=0

#loop for runing 10-times-10-fold cross validation...
#..for different percentage and types of class-label noise
for j in [1,2]: 
    #j=1: Contradictory Examples
    #j=2: Misclassifications
    
    for i in noises:    
        y=k_fold_t_times(ds,trprc=i,tsprc=0,ntype=j)
        acc3[q].append(y[0])
        var3[q].append(y[1])
    q+=1

#Without noise
y=k_fold_t_times(ds)
acc3[q].append(y[0])
var3[q].append(y[1])

#Plotting the figure of Curves
plt.figure(figsize=(8,6))
for i in range(len(acc3)-1): plt.plot(np.multiply(noises,100),acc3[i],'*-', markersize=9)
plt.plot(0,acc3[q],'*', markersize=12)
plt.title('Tic Toc Toe Dataset with Class-Label Noise')
plt.legend(('Contradictory Examples','Misclassifications','Without Noise'), fontsize=11, labelspacing=1, loc='lower left')
plt.xlabel('Noise(%)', fontsize=12)
plt.ylabel('Accuracy(%)', fontsize=12)
plt.xticks(np.arange(0,16,5), fontsize=12)
plt.yticks(np.arange(70,91,5), fontsize=12)
plt.grid()
plt.show()

#Wine Dataset:
noises=[0.05,0.1,0.15]

#acc4=list of average accuracies
acc4=[[],[],[]]

#var4=list of variances
var4=[[],[],[]]
q=0

#loop for runing 10-times-10-fold cross validation...
#..for different percentage and types of class-label noise
for j in [1,2]:
    #j=1: Contradictory Examples
    #j=2: Misclassifications
    
    for i in noises:    
        y=k_fold_t_times_c(dc,trprc=i,tsprc=0,ntype=j)
        acc4[q].append(y[0])
        var4[q].append(y[1])
    q+=1

y=k_fold_t_times_c(dc)
acc4[q].append(y[0])
var4[q].append(y[1])

#Plotting the figure of Curves
plt.figure(figsize=(8,6))
for i in range(len(acc4)-1): plt.plot(np.multiply(noises,100),acc4[i],'*-', markersize=9)
plt.plot(0,acc4[q],'*', markersize=12)
plt.title('Wine Dataset with Class-Label Noise')
plt.legend(('Contradictory Examples','Misclassifications','Without Noise'), fontsize=11, labelspacing=1, loc='lower left')
plt.xlabel('Noise(%)', fontsize=12)
plt.ylabel('Accuracy(%)', fontsize=12)
plt.xticks(np.arange(0,16,5), fontsize=12)
plt.yticks(np.arange(75,101,5), fontsize=12)
plt.grid()
plt.show()