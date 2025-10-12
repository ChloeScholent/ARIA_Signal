import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from sporco.admm import bpdn
from scipy.io import wavfile

# Put the original signal into overlapping frame
def create_frames(x,Nw,Nh):
    X = np.array([ x[i:i+Nw] for i in range(0,len(x)-Nw,Nh) ])
    X=np.transpose(X)
    return X

# Sparse coding based on Hard Thresholding Gradient Descent (matrix version)
def sparse_coding_matrix(X,D,K0):
    niter = 100
    gamma = 1/(np.linalg.norm(D,2)**2)
    Nw,Nd=np.shape(X)
    Nw,K=np.shape(D)
    Z=np.zeros((K,Nd))
    for i in range(niter):
        r=np.dot(D,Z)-X
        C=Z - gamma * np.dot(np.transpose(D),r)
        for j in range(Nd):
            c=C[:,j]
            ind=np.argsort(np.abs(c))
            c[ind[np.arange(0,K-K0)]]=0        
            Z[:,j]=c
    return Z

# Dictionary learning based on Proximal Gradient Descent (matrix version)
def dictionary_learning_matrix(X,Z,D):
    niter = 100
    gamma = 1/(np.linalg.norm(Z,2)**2)
    Nw,Nd=np.shape(X)
    Nw,K=np.shape(D)
    for i in range(niter):
        r=np.dot(D,Z)-X
        D=D - gamma * np.dot(r,np.transpose(Z))
        for j in range(K):
            D[:,j]=D[:,j]/np.sqrt(np.sum(D[:,j]**2))
    return D


sample_rate, audio_data = wavfile.read("dataset/mono_XC77547.wav")
Fs = sample_rate
x = audio_data
N = np.size(x)
t=np.arange(N)/Fs
t_start=t[0]
t_end=t[-1]

print(f"Sampling Frequency : {Fs} Hz")
print(f"Number of samples : {N}")
print(f'Length of the signal : {N/Fs:.2f}s')
print('\n')


# x[n] as a function of time
plt.figure("x[n] as a function of time (2nd signal)")
plt.plot(t,x)
plt.xlim((t[0],t[-1]))
plt.xlabel('Time (s)')
plt.title('Unknown bird song')
plt.show()

Nw=12 # Number of samples per frame
Nh=12 # Hop length
X = create_frames(x,Nw,Nh)
Nw,Nd=np.shape(X)

# Plot of the original signal (in frames)
plt.figure("Plot of the original signal (in frames of length 12)")
for i in range(10):
    plt.subplot(5,2,i+1)
    plt.plot(X[:,i])
plt.show()


# Generate random dictionary
K = 3 # Number of atoms
D=np.random.randn(Nw,K)
for k in range(K):
    D[:,k]=(D[:,k]-np.mean(D[:,k]))
    D[:,k]=D[:,k]/np.sqrt(np.sum(D[:,k]**2))

# Plot of the initialized dictionary
plt.figure("Plot of the initialized random dictionary")
for i in range(K):
    plt.subplot(K,1,i+1)
    plt.plot(D[:,i])
plt.show()

# Perform the alternate minimization (sparse coding + dictionary learning)
K0=1
for i in range(50):
    Z=sparse_coding_matrix(X,D,K0)
    D=dictionary_learning_matrix(X,Z,D)

# Plot the learned atoms and activations
Z_max=np.max(Z)
Z_min=np.min(Z)
D_max=np.max(D)
D_min=np.min(D)
plt.figure("Learned atoms and activations")
for k in range(K):
    plt.subplot(K,4,4*k+1)
    plt.plot(D[:,k])
    plt.xlim((0, Nw))
    plt.ylim((D_min, D_max))
    plt.subplot(K,4,(4*k+2,4*k+4))
    plt.bar(np.arange(Nd),Z[k,:])
    plt.ylim((Z_min,Z_max))
plt.show()


# Approximated signal with dictionary learning
X_bar=np.dot(D,Z)
y=icreate_frames(X_bar,Nh)

plt.figure("Reconstructed signal with dictionary learning")
plt.plot(x)
plt.plot(y)
plt.legend(('Original signal', 'Reconstructed signal'))
plt.show()



