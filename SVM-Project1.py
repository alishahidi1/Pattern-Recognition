import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score
from sklearn.svm import*
from time import*

#loading dataset1 & dataset2
ds1=np.array([[1.9643,4.5957,1.]]) #there is a problem in the first row numbers of 
                                    #the csv file; it is added manually here
ds1=np.append(ds1,np.loadtxt('hw3_dataset1.csv', delimiter=',',skiprows=1),axis=0)


ds2=np.array([[-0.158986,0.423977,1.]]) #there is a problem in the first row numbers of 
                                        #the csv file; it is added manually here
ds2=np.append(ds2,np.loadtxt('hw3_dataset2.csv', delimiter=',',skiprows=1),axis=0)

#r is the list of values of C for the LinearSVM Classifier
r=[0.001,0.01,0.1,1]

#aij are the i'th coefficient of the LinearSVM classifier
#for the j'th dataset
a01=[]
a11=[]
a21=[]
a02=[]
a12=[]
a22=[]

#acci is the list of accuracies of the trained classifiers for
#dataset #i
acc1=[]
acc2=[]
#Calculating the Mean & Standard deviation of each dataset
#for normalization; mui:mean of dataset i, di: standard deviation
#of dataset i

mu1=np.mean(ds1[:,0:2], axis=0)
d1=np.std(ds1[:,0:2], axis=0)#-np.min(ds1[:,0:2], axis=0)

mu2=np.mean(ds2[:,0:2], axis=0)
d2=np.std(ds2[:,0:2], axis=0)#-np.min(ds2[:,0:2], axis=0)

#Xni is the normalized dataset #i
Xn1=(ds1[:,0:2]-mu1)/d1
Xn2=(ds2[:,0:2]-mu2)/d2

#For different values of C, LinearSVM classifer is trained
for i in range(len(r)):
    
    #lsvmi is the LinearSVM classifier defined for dataset #i
    lsvm1 = SVC(kernel='linear',C=r[i])#,tol=1e-5, max_iter=1e5)
    lsvm2 = SVC(kernel='linear',C=r[i])#,tol=1e-5, max_iter=1e5)
    
    #Training the LinearSVM by fitting feature values of 
    #datasets
    lsvm1.fit(Xn1,ds1[:,2])
    lsvm2.fit(Xn2,ds2[:,2])
    
    #Obtaining coefficients of the trained classifier and the 
    #corresponding accuracy
    a11.append(lsvm1.coef_[0,0])
    a21.append(lsvm1.coef_[0,1])
    a01.append(lsvm1.intercept_[0])
    acc1.append(accuracy_score(ds1[:,2],lsvm1.predict(Xn1)))
    
    a02.append(lsvm2.intercept_[0])
    a12.append(lsvm2.coef_[0,0])
    a22.append(lsvm2.coef_[0,1])
    acc2.append(accuracy_score(ds2[:,2],lsvm2.predict(Xn2)))

#plotting the dataset1 points, decision boundaries and margins
#defining subplots for different values of C
fig,ax1=plt.subplots(2,2,figsize=(12,12))

q=0
#for each values of C, decision boundaries and margins are calculated and 
#ploted in 1 subplot
for i in range(2):
    for j in range(2):

        #plotting the points
        p1=ax1[i,j].scatter(ds1[:,0],ds1[:,1],c=ds1[:,2])
        x1=np.arange(min(Xn1[:,0])-1,max(Xn1[:,0])+1,0.1)
        
        #obtaining the margin lines and the decision boundary
        aa=-a11[q]/a21[q]
        mrg=1/np.sqrt(a11[q]**2+a21[q]**2)
        y_b=-a11[q]/a21[q]*x1-a01[q]/a21[q]
        #plotting the decision boundary
        ax1[i,j].plot(x1*d1[0]+mu1[0],y_b*d1[1]+mu1[1],label='Boundary')#'C='+str(r[i]))

        #plotting the lower margin line
        ax1[i,j].plot(x1*d1[0]+mu1[0],(y_b-np.sqrt(1+aa**2)*mrg)*d1[1]+mu1[1],'k--',label='Margin')
        
        #plotting the upper margin line
        ax1[i,j].plot(x1*d1[0]+mu1[0],(y_b+np.sqrt(1+aa**2)*mrg)*d1[1]+mu1[1],'k--')


        ax1[i,j].add_artist(ax1[i,j].legend(*p1.legend_elements(), loc="lower left", title="Classes"))
        ax1[i,j].legend(loc='upper right')
        ax1[i,j].set_title('C='+str(r[q]))
        ax1[i,j].set_xlim([-0.5,4.5])
        q+=1

ax1[0,0].set_ylabel('X2')
ax1[1,0].set_ylabel('X2')
ax1[1,0].set_xlabel('X1')
ax1[1,1].set_xlabel('X1')
plt.show()
print('Accuracy1=',acc1)


#plotting the dataset2 points, decision boundaries and margins
#defining subplots for different values of C
fig,ax2=plt.subplots(2,2,figsize=(12,12))

q=0
for i in range(2):
    for j in range(2):

        #plotting the points
        p2=ax2[i,j].scatter(ds2[:,0],ds2[:,1],c=ds2[:,2])
        x2=np.arange(min(Xn2[:,0])-2,max(Xn2[:,0])+2,0.1)

        #obtaining the margin lines and the decision boundary
        aa=-a12[q]/a22[q]
        mrg=1/np.sqrt(a12[q]**2+a22[q]**2)
        y_b=-a12[q]/a22[q]*x2-a02[q]/a22[q]
        
        #plotting the decision boundary
        ax2[i,j].plot(x2*d2[0]+mu2[0],y_b*d2[1]+mu2[1],label='Boundary')

        #plotting the lower margin line
        ax2[i,j].plot(x2*d2[0]+mu2[0],(y_b-np.sqrt(1+aa**2)*mrg)*d2[1]+mu2[1],'k--',label='Margin')
        
        #plotting the upper margin line
        ax2[i,j].plot(x2*d2[0]+mu2[0],(y_b+np.sqrt(1+aa**2)*mrg)*d2[1]+mu2[1],'k--')


        ax2[i,j].add_artist(ax2[i,j].legend(*p2.legend_elements(), loc="lower right", title="Classes"))
        ax2[i,j].legend(loc='upper left')
        ax2[i,j].set_title('C='+str(r[q]))
        ax2[i,j].set_xlim([-0.7,0.4])
        q+=1

ax2[0,0].set_ylabel('X2')
ax2[1,0].set_ylabel('X2')
ax2[1,0].set_xlabel('X1')
ax2[1,1].set_xlabel('X1')

ax2[0,0].set_ylim([-1,3])
ax2[0,1].set_ylim([-1,1])
ax2[1,0].set_ylim([-1,1])
ax2[1,1].set_ylim([-1,1])
plt.show()
print('Accuracy2=',acc2)