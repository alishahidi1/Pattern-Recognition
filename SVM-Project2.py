import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score
from sklearn.svm import*
from time import*

#defining a function for splitting samples to a training set and
#a test set
def split(d,k,i):
    #d=dataset, k=number of folds, i=index of test fold
    n=len(d)//k
    if i<k-1:
        d_test=d[i*n:i*n+n]
        d_train=np.delete(d, np.s_[i*n:i*n+n],axis=0)
    else:
        d_test=d[i*n:]
        d_train=d[:i*n]
    #d_train=training set
    #d_test=test set
    return d_train, d_test

#Defining a function for calculating
#the accuracy of Linear SVM classifier
def Accuracy(train,test,c):
    
    #train=training set
    #test=test set
    #c=C value
    
    #Defining a Linear SVM Classsifier
    lsvm=SVC(kernel='linear',C=c)#, max_iter=1e5)
    
    #Training the classifier by fitting the 
    #training set
    lsvm.fit(train[:,0:2],train[:,2])
    
    #Predicting the class label of test set using the
    #trained classifier
    predicted=lsvm.predict(test[:,0:2])
    
    N=len(test)
    
    #Obtaining class labels 
    v=np.unique(train[:,-1])
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
        pr=predicted[i]
        cm[dic[tr],dic[pr]]+=1
        
    #this function has 3 outputs:
    #1:Accuracy(%)
    #2: Confusion Matrix
    #3: Linear SVM classifier used for classification
    return np.trace(cm)/N*100,cm,lsvm


#Defining a function for 10-fold-10times Cross Validation
def k_fold_t_times(d,k=10,t=10,c=1):
    #d=dataset, k=number of folds, t= number of times
    #c= C value  
        
    #acc=list of accuracies
    acc=np.zeros(k*t)
    N=len(d)
    n=k*t
    
    #best_acc: best achieved accuracy
    best_acc=0.0
    q=0

    
    #loop for times
    for i in range(t):
        
        #data set is shuffled before doing k-fold cross validation..
        #each time
        
        d_new=d[np.random.permutation(N)]
        
        #loop for folds
        for j in range(k):
            #splitting data samples to the training & test sets
            d_split=split(d_new,k,j)
            
            d_train=d_split[0]
            d_test=d_split[1]
            
            #the dataset will be normalized by making the mean of the
            #feature values equal to zero and making their standard deviation
            #equal to 1
            
            #mu=mean of feature values of training set
            #sd=standard deviation of feature values of training set
            mu=np.mean(d_train[:,0:2],axis=0)
            sd=np.std(d_train[:,0:2], axis=0)
            
            #normalizing the training set
            d_train=np.insert((d_train[:,0:2]-mu)/sd,2,d_train[:,2],axis=1)
            
            #normalizing the test set
            d_test=np.insert((d_test[:,0:2]-mu)/sd,2,d_test[:,2],axis=1)
            
            #Calculating the accuracy
            acrs=Accuracy(d_train,d_test, c)
            
            #acc=array of all accuracies
            acc[q]=acrs[0]
            
            #Calculating the confusion matrix with respect to..
            #... the maximum accuracy
            if acc[q]>best_acc: 
                best_acc=acc[q]
                cm=acrs[1]
                best_lsvm=acrs[2]
                best_mu=mu
                best_sd=sd
            q+=1
    
    #this function has 7 outputs:
        #1:average accuracy(%)
        #2:variance of the accuracy
        #3: Confusion Matrix with respect to best accuracy
        #4: the best achieved accuracy
        #5: Linear SVM classifier with respect to best accuracy
        #6: mean of feature values of the dataset with best accuracy
        #7: standard deviation of feature values of the dataset with best accuracy
    return np.mean(acc),np.var(acc)*1e-4,cm,best_acc,best_lsvm,best_mu,best_sd


#adabM1: a function for creating an Adaboost-M1 classifier
def adabM1(d,c=1,T=50,ns=100):
    #d= dataset
    #c= C value
    #T= the number of weak learners
    #ns= number of selected samples for training each classifier
    
    #dd is a copy of the original dataset
    dd=d.copy()
    N=len(dd)
    
    #pr=array of sample weights; initialized here with equal values
    pr=np.array([1/N for i in range(N)])
    
    #svm_cl=list of Linear SVM classifiers
    svm_cl=[]
    
    #beta_t=list of beta_t values with respect to the classifier
    beta_t=[]
    
    #i=index of the currently training classifier
    i=0
    
    #training T unique Linear SVM classifiers
    while i<T:
        
        #ind=indices of randomly selected classifiers using
        #their weights
        ind=np.random.choice(N,ns, p=pr, replace=False)

        #lsvm: Linear SVM classifier
        lsvm=SVC(kernel='linear',C=c)#, max_iter=1e5)
        
        #lsvm is trained by fitting the selected samples 
        lsvm.fit(dd[ind,0:2],dd[ind,2])
        
        #predicted = The predicted class labels of all samples
        #by using the trained Linear SVM classifier
        predicted=lsvm.predict(dd[:,0:2])
        
        #e_k= hypothesis error
        #ind_w=indices of miscalssified samples
        #ind_r=indices of correctly classified samples
        e_k=0
        ind_w=predicted!=dd[:,2]
        ind_r=predicted==dd[:,2]
        e_k=sum(pr[ind_w])

        #if training error is less than 50%, consider the classifier
        #otherwise, discard it and train another classifier after
        #resampling
        if e_k<0.5:
            
            #calculating beta coefficients of the classifier
            betat=e_k/(1-e_k)
            
            #updating the weights
            pr[ind_r]=pr[ind_r]*betat
            pr=pr/sum(pr)
            
            beta_t.append(betat)
            svm_cl.append(lsvm)
            i+=1
    
    #this function has 2 outputs:
        #1. List of beta coefficients
        #2. List of Linear SVM classifiers
    
    return beta_t,svm_cl

#pr_adabM1 a function for predicting class labels of 
# a dataset by using the list of beta and corresponding
# Linear SVM classifiers
def pr_adabM1(X,beta_t,svm_cl):#,mu,dr):
    #X=dataset, beta_t=list of beta coffecients
    #svm_cl=list of Linear SVM classifiers
    
    c=np.log(1/np.array(beta_t))
    p=list(map(lambda x:x.predict(X),svm_cl))
    h=np.dot(c,p)
    
    #here, the labels of classes A & B are -1 & +1
    #respectively, therefore, we can use the "sign" function
    return np.sign(h)


#Accuracy_adabM1: a function for calculating
#the accuracy of an Adaboost-M1 classifier
def Accuracy_adabM1(train,test,c=1,T=50,ns=100):
    
    #train=training set
    #test=test set
    #c=C value
    #T= the number of weak learners
    #ns= number of selected samples for training each classifier
    
    #adab=a trained ensemble of Linear SVM classifiers based on
    #the Adaboost-M1 approach
    adab=adabM1(train,c,T,ns)
    
    #beta_t=list of beta coefficients
    beta_t=adab[0]
    
    #svm_cl=list of trained Linear SVM classifiers
    svm_cl=adab[1]
        
    N=len(test)
    
    #Obtaining Class labels
    v=np.unique(train[:,-1])
    m=len(v)
    dic={}
    
    #constructing the confusion matrix
    for i in range(m):
        dic[v[i]]=i
    
    #cm=Confusion Matrix
    cm=np.zeros((m,m), dtype=int)
    
    #predicted labels are compares with the actual labels..
    #then the accuracy and the confusion matrix are obtained
    
    #Predicting the class label of test set using the
    #trained classifier
    predicted=pr_adabM1(test[:,0:2],beta_t,svm_cl)
    
    #predicted labels are compares with the actual labels..
    #then the accuracy and the confusion matrix are obtained
    for i in range(N):
            tr=test[i,-1]
            pr=predicted[i]
            cm[dic[tr],dic[pr]]+=1
    
    #this function has 3 outputs:
        #1:Accuracy(%)
        #2: Confusion Matrix
        #3: the trained Adaboost-M1 classifier
    return np.trace(cm)/N*100,cm,adab

#k_fold_t_times_adabM1: a function for 10-fold-10times Cross Validation
# for Adaboost-M1 approach
def k_fold_t_times_adabM1(d,k=10,t=10,c=1,T=50,ns=100):
    #d=dataset, k=number of folds, t= number of times
    #c=C value
    #T= the number of weak learners
    #ns= number of selected samples for training each classifier   
        
    #acc=list of accuracies
    acc=np.zeros(k*t)
    N=len(d)
    n=k*t
    
    #best_acc: best achieved accuracy
    best_acc=0.0
    q=0

    
    #loop for times
    for i in range(t):
        
        #data set is shuffled before doing k-fold cross validation..
        #each time
        
        d_new=d[np.random.permutation(N)]
        
        #loop for folds
        for j in range(k):
            #splitting data samples to the training & test sets
            d_split=split(d_new,k,j)
            
            d_train=d_split[0]
            d_test=d_split[1]
            
            #the dataset will be normalized by making the mean of the
            #feature values equal to zero and making their standard deviation
            #equal to 1
            
            #mu=mean of feature values of the training set
            #sd=standard deviation of feature values of the training set
            mu=np.mean(d_train[:,0:2], axis=0)
            sd=np.std(d_train[:,0:2], axis=0)
            
            #normalizing the training set
            d_train=np.insert((d_train[:,0:2]-mu)/sd,2,d_train[:,2],axis=1)
            
            #normalizing the test set
            d_test=np.insert((d_test[:,0:2]-mu)/sd,2,d_test[:,2],axis=1)

            
            #Calculating the accuracy
            acrs=Accuracy_adabM1(d_train,d_test, c,T,ns)
            
            #acc=array of all accuracies
            acc[q]=acrs[0]

            #Calculating the confusion matrix with respect to..
            #... the maximum accuracy
            if acc[q]>best_acc: 
                best_acc=acc[q]
                cm=acrs[1]
                best_adab=acrs[2]
                best_mu=mu
                best_std=sd
            q+=1
    
    #this function has 7 outputs:
        #1: average accuracy(%)
        #2: variance of the accuracy
        #3: Confusion Matrix with respect to the best accuracy
        #4: the best achieved accuracy
        #5: Adaboost-M1 classifier with respect to the best accuracy
        #6: mean of feature values of the dataset with the best accuracy
        #7: standard deviation of feature values of the dataset with the best accuracy
    return np.mean(acc),np.var(acc)*1e-4,cm,best_acc,best_adab,best_mu,best_std

#Loading ClassA & ClassB datasets
#clA: dataset with 'A' label
#clB: dataset with 'B' label

clA=np.loadtxt('classA.csv', delimiter=',')
clB=np.loadtxt('classB.csv', delimiter=',')

#db: dataset after combining Class A & Class B samples
db=np.append(np.insert(clA,2,-1,axis=1),np.insert(clB,2,1,axis=1),axis=0)

#r is the list of values of C for the LinearSVM Classifier
r=[0.1,1,10,100]

#a01, a02, and a03 are the coefficients of the LinearSVM classifier
a03=[]
a13=[]
a23=[]

#acc_C: the list of average accuracies for different values of C
acc_C=[]

#var_C: the list of the variances of the accuracy for different values of C
var_C=[]

#lsvm= List of Linear SVM classifiers
lsvm=[]

#mu= List of mean values
mu=[]

#sd= List of standard deviation values
sd=[]

#for each value of C, the decision boundary of the
#Linear SVM classifier is achieved
for i in range(len(r)):
    
    #performing cross validation for obtaining 
    #the Linear SVM classifier with the maximum
    #Accuracy for each value of C
    cv=k_fold_t_times(db,c=r[i])
    
    #cv[-3]:Linear SVM classifier with the maximum accuracy
    lsvm.append(cv[-3])
    
    #cv[-2]: Mean of the feature values with respect to the maximum accuracy
    mu.append(cv[-2])
    
    #cv[-1]: Standard Deviation of the feature values with respect to
    #Maximum Accuracy
    sd.append(cv[-1])
    
    acc_C.append(cv[0])
    var_C.append(cv[1])
    
    #Xn3 is the normalized form of dataset
    Xn3=(db[:,0:2]-mu[i])/sd[i]
    
    #Obtainig coefficients of the classifier with
    #the maximum accuracy
    a03.append(lsvm[i].intercept_[0])
    a13.append(lsvm[i].coef_[0,0])
    a23.append(lsvm[i].coef_[0,1])
    

#performing 10-fold-10-times cross validation for
#obtaining an ensemble of Linear SVM classifiers by 
#Adaboost-M1 approach
cv_en=k_fold_t_times_adabM1(db, c=1, T=50, ns=100)

#acc_adab= average accuracy of the Adaboost-M1 classifier
acc_adab=cv_en[0]

#var_adab= variance of the accuracy of the Adaboost-M1 classifier
var_adab=cv_en[1]

#adab_en: Adaboost-M1 classifier with the maximum accuracy
adab_en=cv_en[-3]

#mu_en: Mean of the feature values with respect to the maximum accuracy
mu_en=cv_en[-2]

#sd_en: Standard Deviation of the feature values with respect to
#Maximum Accuracy
sd_en=cv_en[-1]

#dn: the normalized form of dataset using mu_en & sd_en
dn=np.insert((db[:,0:2]-mu_en)/sd_en,2,db[:,2],axis=1)

#x1r & x2r are used to form the x & y axes of the figure
x1r=np.arange(180,450,0.5)
x2r=np.arange(0,350,0.5)

#x1n & x2n are the normalized forms of x1r & x2r respectively
x1n=(x1r-mu_en[0])/sd_en[0]
x2n=(x2r-mu_en[1])/sd_en[1]

#making a meshgrid
x1r, x2r = np.meshgrid(x1r,x2r)
m=len(x1n)
n=len(x2n)

#z: the array of predicted class label of all of the points
#in the figure plane using Adaboost-M1 approach
z = np.array([[0 for i in range(m)] for j in range(n)])

#Predicting class-label of all of the points in the figure plane
#by the Adaboost-M1 classifying method
for i in range(n):
    X=np.insert([x1n],1,x2n[i],axis=0).transpose()
    z[i,:]=pr_adabM1(X,adab_en[0],adab_en[1])

#Plotting dataset points and decision boundaries    
plt.figure(figsize=(11,10))
ax3=plt.gca()

#plotting decision area of the Adaboost-M1 classifier
ax3.plot(x1r[z==-1],x2r[z==-1],'s',c=(1,0.6,0.6))
ax3.plot(x1r[z==1],x2r[z==1],'s',c=(0.7,0.7,1))#,cmap=cm.bwr_r)
ax3.add_artist(ax3.legend(('A','B'),loc='lower left',title='Predicted Class-label'+'\n'+'Adaboost-M1',
                        framealpha=1, fontsize=12,title_fontsize=12))

#plotting the decision boundaries for different values of C
x3=np.arange(min(Xn3[:,0]),max(Xn3[:,0]),0.1)
for i in range(len(r)): 
    ax3.plot(x3*sd[i][0]+mu[i][0],(a13[i]*x3+a03[i])/(-a23[i])*sd[i][1]+mu[i][1],label='C='+str(r[i]))
ax3.add_artist(ax3.legend(loc='upper left', title='Decision Boundaries', framealpha=1, 
          fontsize=12, title_fontsize=12))

#plotting the dataset1 points
p1,=ax3.plot(clA[:,0],clA[:,1],'*r', label='Class A')
p2,=ax3.plot(clB[:,0],clB[:,1],'*b', label='Class B')
ax3.add_artist(ax3.legend(handles=[p1,p2],title='Samples',loc='upper right',
                        framealpha=1, fontsize=12,title_fontsize=12))

plt.margins(0)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
print('Mean of the accuracy for different values of C =',acc_C,'\n')
print('Variance of the accuracy for different values of C =',var_C,'\n')
print('Mean of the accuracy for the Adaboost-M1 classifier =',acc_adab,'\n')
print('Variance of the accuracy for the Adaboost-M1 classifier =',var_adab,'\n')



#plotting the points, decision boundaries, and margin lines
#defining subplots for different values of C
fig,ax3=plt.subplots(2,2,figsize=(12,12))

q=0

#for each values of C, decision boundaries and margins are calculated and 
#ploted in 1 subplot
for i in range(2):
    for j in range(2):

        x3=np.arange(min(Xn3[:,0]),max(Xn3[:,0]),0.1)

        #obtaining the margin lines and the decision boundary 
        aa=-a13[q]/a23[q]
        mrg=1/np.sqrt(a13[q]**2+a23[q]**2)

        #plotting the decision boundary
        ax3[i,j].plot(x3*sd[q][0]+mu[q][0],(a13[q]*x3+a03[q])/(-a23[q])*sd[q][1]+mu[q][1],label='Boundary')
        
        #plotting the lower margin line
        ax3[i,j].plot(x3*sd[q][0]+mu[q][0],((a13[q]*x3+a03[q])/(-a23[q])-np.sqrt(1+aa**2)*mrg)*sd[q][1]+mu[q][1],'k--',label='Margin')
        
        #plotting the upper margin line
        ax3[i,j].plot(x3*sd[q][0]+mu[q][0],((a13[q]*x3+a03[q])/(-a23[q])+np.sqrt(1+aa**2)*mrg)*sd[q][1]+mu[q][1],'k--')
        ax3[i,j].add_artist(ax3[i,j].legend(loc='upper left'))

        #plotting the points
        p1,=ax3[i,j].plot(clA[:,0],clA[:,1],'*r', label='Class A')
        p2,=ax3[i,j].plot(clB[:,0],clB[:,1],'*b', label='Class B')
        ax3[i,j].add_artist(ax3[i,j].legend(handles=[p1,p2],title='Samples',loc='lower left'))

        ax3[i,j].set_title('C='+str(r[q]))
        ax3[i,j].set_xlim([180,450])
        q+=1

ax3[0,0].set_ylabel('X2')
ax3[1,0].set_ylabel('X2')
ax3[1,0].set_xlabel('X1')
ax3[1,1].set_xlabel('X1')

plt.show()