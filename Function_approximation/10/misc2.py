# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 06:16:03 2019

@author: jairam
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:48:43 2019

@author: jairam
"""


import numpy as np
import sys
from math import sqrt
from random import seed
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train_validate_test_split(df, train_percent=.7, validate_percent=.1):
    
    split_1 = int(0.7 * len(df))
    split_2 = int(0.8 * len(df))
    dataset_train = df[:split_1]
    dataset_val = df[split_1:split_2]
    dataset_test = df[split_2:]
    return dataset_train,dataset_val,dataset_test


# =============================================================================

# =============================================================================

#output layer actiavtion fns


def softmax(x, beta,derivative=False):
    if (derivative == True):
        return beta*x * (1 - x)
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

#hiiden layer activation functions

def tanh_fn(x):
    return ( np.exp(x) - np.exp(-x)) / ( np.exp(x) + np.exp(-x))

def tanh(x,beta, derivative=False):
    if (derivative == True):
        return beta*(1 - (x ** 2))
    return tanh_fn(beta*x)

def softplus(x, beta,derivative=False):
    if(derivative==True):
        return 1 / (1 + np.exp(-x))
    return np.log(1+np.exp(x))

def elu(x,delta, derivative=False):
    if(derivative==True):
        if(x>0):
            return 1
        else :
            return delta*np.exp(x)
    if(x>0):
        return x
    else :
        return delta*(np.exp(x)-1)
    

def relu(x, derivative=False):
    if(derivative==True):
        return np.where(x>0,1,0)
    return np.where(x>0,x,0)

# =============================================================================
def sigmoid(x, beta, derivative=False):
    if (derivative == True):
        return beta*x * (1 - x)
    return 1 / (1 + np.exp(-beta*x))
# =============================================================================

    

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax, normalize = False):
    if normalize:
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
            

filename='housing-data.txt'
dataset=list()
f=open(filename,'r')
for line in f.readlines():
    dataset.append(np.array([float(x) for x in line.split()[0:14]]))
f.close()    
dataset=np.array(dataset)
dataset=dataset.tolist()


# =============================================================================
# minmax=dataset_minmax(dataset)
# normalize_dataset(dataset, minmax)
# 
# =============================================================================
print('Enter True for Normalized or enter False for unnormalized. ')
s1 = input()

print('Enter the activation functions: \t 1.Logistic \t 2.Tanh \t 3.ReLU \t 4.Softplus \t 5.ELU')
s2 = input()

print('Enter learning mode \t 1.Pattern \t 2.Batch')
s3 = input()

print('Enter weight update rule \t 1.Delta \t 2.Generalized delta \t 3.AdaGrad \t 4.RMSProp \t 5.AdaDelta \t 6.Adam')
s4 = input()

minmax=dataset_minmax(dataset)
normalize_dataset(dataset, minmax,bool(int(s1)))

dataset=np.asarray(dataset)
# =============================================================================
# X=dataset[:,0:13]
# y=dataset[:,13]
# =============================================================================
# =============================================================================
# net=initialize_network(dataset)
# X=dataset[:,0:13]
# y=dataset[:,13]
# print(net)
# l_rate=0.2
# n_epochs=1500
# n_outputs=1
# errors=training(net,n_epochs,l_rate,n_outputs,X,y)
# =============================================================================

#shuffling dataset
#dataset = np.concatenate((X,y),axis = 1)
np.random.seed(8)
np.random.shuffle(dataset)
df=dataset.tolist()
dataset_train,dataset_val,dataset_test=train_validate_test_split(df,0.7,0.1)
#train
dataset_train=np.asarray(dataset_train)
X_train = dataset_train[:,0:13]
y_train = dataset_train[:,13:]
#val
dataset_val=np.asarray(dataset_val)
X_val = dataset_val[:,0:13]
y_val = dataset_val[:,13:]
#test
dataset_test=np.asarray(dataset_test)
X_test = dataset_test[:,0:13]
y_test = dataset_test[:,13:]


def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])  #logp=N*1
    loss = np.sum(logp)  #computed for all examples(N)
    return loss

class MyNN:
    def __init__(self, x, y,neurons1,neurons2,seed_no):
        self.x = x
        self.lr = 0.00001
        ip_dim = x.shape[1]
        op_dim = y.shape[1]
        
        self.wIJ = np.random.randn(ip_dim, neurons1)#between input,first hidden(d*neurons1)
        self.bh1 = np.zeros((1, neurons1))#1*neurons1
# =============================================================================
        self.wJM = np.random.randn(neurons1, neurons2)#between 2nd hidden,first hidden(neurons1*n2)
        self.bh2 = np.zeros((1, neurons2))#1*n2
# =============================================================================
        self.wMK= np.random.randn(neurons2, op_dim)#between 2nd hidden,output(n2*K)
        self.bo = np.zeros((1, op_dim))#1*K
        self.y = y#N*K
        self.cs = dict()
        self.cs['normalize'] = dict()
        self.cs['normalize'] = {'True':True,'False':False}
        self.cs['activation function'] = dict()
        self.cs['activation function'] = {'Logistic':sigmoid, 'Tanh':tanh, 'ReLU':relu, 'Softplus':softplus, 'ELU':elu}
        self.cs['activation function derivative'] = {'Logistic':sigmoid, 'Tanh':tanh, 'ReLU':relu, 'Softplus':softplus, 'ELU':elu}
        #self.cs['Learning mode'] = dict()
        #cs['Learning mode'] = {'Pattern':pattern(), 'Batch':batch()}
        self.cs['Pattern'] = dict()
        self.cs['Batch'] = dict()
        self.cs['Pattern']['feedforward'] = self.feedforward_pattern
        self.cs['Batch']['feedforward'] = self.feedforward_batch
        self.cs['Pattern']['Weight update'] = dict()
        self.cs['Batch']['Weight update'] = dict()        
        self.cs['Pattern']['Weight update'] = {'Delta':self.delta_pattern}
        self.cs['Batch']['Weight update'] = {'Delta':self.delta_batch}
        self.cs['Loss functions'] = dict()


    def feedforward_pattern(self,n,beta):
        #for first hidden layer
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = self.cs['activation function'][s2](ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = self.cs['activation function'][s2](ah2, beta)#1*nuerons2
# =============================================================================
        #for last layer
        ao = np.dot(self.sh2, self.wMK) + self.bo
        self.so = self.cs['activation function']["ReLU"](ao)#for output layer  #1*K



        
    def feedforward_batch(self,rnum,beta):

        ah1 = np.dot(self.x, self.wIJ) + self.bh1
        self.sh1 = sigmoid(ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = sigmoid(ah2,beta)#1*nuerons2
# =============================================================================
        #for last layer
        ao = np.dot(self.sh2, self.wMK) + self.bo
        self.so = relu(ao)#for output layer  #1*K




        
    def delta_pattern(self,n,beta):
        #pattern mode
        #update of wmk
        x_n=self.x[n,:]
        x_n=x_n[np.newaxis,:]  #1*d
        y_n=self.y[n,:]
        y_n=y_n[np.newaxis,:]
        
        
        z3_delta = self.so - y_n # w3
        a3_delta = z3_delta*self.cs['activation function derivative']['ReLU'](self.so,derivative = True) 
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * self.cs['activation function derivative'][s2](self.sh2,beta,derivative = True) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * self.cs['activation function derivative'][s2](self.sh1,beta, derivative = True) # w1
 
        self.wMK -= self.lr * np.dot(self.sh2.T, a3_delta)
        self.bo -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        self.wJM -= self.lr * np.dot(self.sh1.T, a2_delta)
        self.bh2 -= self.lr * np.sum(a2_delta, axis=0)
# =============================================================================
        self.wIJ -= self.lr * np.dot(x_n.T, a1_delta)
        self.bh1 -= self.lr * np.sum(a1_delta, axis=0)       

        



    def delta_batch(self,rnum,beta):
       
        
        z3_delta = (self.so - self.y)/self.x.shape[0] # w3
        a3_delta = z3_delta*relu(self.so,derivative=True)
# =============================================================================
        z2_delta = np.dot(a3_delta, self.wMK.T)
        a2_delta = z2_delta * sigmoid_derv(self.sh2,beta) # w2
# =============================================================================
        z1_delta = np.dot(a2_delta, self.wJM.T)
        a1_delta = z1_delta * sigmoid_derv(self.sh1,beta) # w1
 
        self.wMK -= self.lr * np.dot(self.sh2.T, a3_delta)
        
        #print(self.wMK[0][0])
        self.bo -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
# =============================================================================
        self.wJM -= self.lr * np.dot(self.sh1.T, a2_delta)
        self.bh2 -= self.lr * np.sum(a2_delta, axis=0)
# =============================================================================
        self.wIJ -= self.lr * np.dot(self.x.T, a1_delta)
        self.bh1 -= self.lr * np.sum(a1_delta, axis=0)       



    def predict_feedforward(self,xx,yy,beta):
        x_n = xx[np.newaxis,:]
        ah1 = np.dot(x_n, self.wIJ) + self.bh1
        self.sh1 = self.cs['activation function'][s2](ah1,beta)#for first hidden layer
        #2nd hidden layer
# =============================================================================
        ah2 = np.dot(self.sh1, self.wJM) + self.bh2
        self.sh2 = self.cs['activation function'][s2](ah2, beta)#1*nuerons2
# =============================================================================
        #for last layer
        ao = np.dot(self.sh2, self.wMK) + self.bo
        self.so = self.cs['activation function']["ReLU"](ao)#for output layer  #1*K    
        mse = (0.5*np.sum(yy-np.array(self.so))**2)
        predicted_val = self.so 
#        true_cls = yy.argmax()
#        pred_cls = self.so.argmax()
        return (mse ,predicted_val) 
        
# =============================================================================
#     def predict(self, data):
#         self.x = data
#         self.predict_feedforward()
#         return self.so.argmax()
#     
#         
# =============================================================================
def get_rmse(x, y,beta):
    sum_error=0
    y_pred_list = list()
    for xx,yy in zip(x, y):
        s = model.predict_feedforward(xx,yy,beta)
        sum_error+=s[0]
        y_pred_list.append(s[1])
#    sum_error = sum_error/(x.shape[0])
    return (sum_error/x.shape[0],np.array(y_pred_list))

best_beta=0
best_neurons=0	
min_val_rmse=sys.maxsize
#check for convergence
seed_no = 8
beta = 1
neurons1 = 30
neurons2 = 3


model = MyNN(X_train, np.array(y_train),neurons1,neurons2,seed_no)
sum_prev_error=0
n_epochs=0        
for j in range(3000):
    sum_error=0
    threshold=0.00001
    if s3 == 'Pattern':
        for n in range(X_train.shape[0]):  #pattern mode
            model.cs[s3]['feedforward'](n,beta)
            model.cs[s3]['Weight update'][s4](n,beta)
    if s3 == 'Batch':
        model.cs[s3]['feedforward'](n,beta)
        model.cs[s3]['Weight update'][s4](n,beta)
    for xx,yy in zip(X_train, np.array(y_train)):
        sum_error+= model.predict_feedforward(xx,yy,beta)[0]   
    #convergence criterion
    
    sum_error = sum_error/X_train.shape[0]
    if(abs(sum_error-sum_prev_error)<=threshold):
        print('parameters : beta={} and neurons1={} and neurons2 = {}'.format(beta,neurons1, neurons2))
        print('convergence has reached with difference of total error=',sum_error-sum_prev_error)
        print('no of epochs for convergence=',n_epochs)
        print("**********")
        #break
    print(abs(sum_error-sum_prev_error))
    sum_prev_error=sum_error
    n_epochs+=1

#            print(sum_error) 
    
    
train_tpl = get_rmse(X_train, y_train,beta)
print("Mean squared error on training data: ",train_tpl[0])
test_tpl=get_rmse(X_test, y_test,beta)
print("Mean squared error for testdata: ", test_tpl[0]) 
print("\n\n")  
print('-----------------------------------------------------------')



    
train_pred = train_tpl[1]
test_pred = test_tpl[1] 

fig = plt.figure(figsize = (20,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.title.set_text('Model output vs desired output for Training dataset')
ax2.title.set_text('Model output vs desired output for Test dataset')

ax1.scatter(train_pred,y_train)
ax2.scatter(test_pred,y_test)
print("\n\n")  
print('-----------------------------------------------------------')

