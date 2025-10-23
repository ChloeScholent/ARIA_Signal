import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Function to compute the relative energy in Nb frequency band
def relative_energy(x,Nb,Fs):
    N=np.size(x)
    #Computation of the FFT
    X=np.fft.fft(x)
    f=np.fft.fftfreq(N, d=1/Fs)
    FNyq=Fs/2
    E=np.zeros((Nb,))
    for i in range(Nb):
        E[i]=np.sum(np.abs(X[(f>i*FNyq/Nb) & (f<=(i+1)*FNyq/Nb)])**2)
    E=E/np.sum(np.abs(X[(f>0) & (f<=Fs/2)])**2)
    return E

def compute_features(X):
    N,M=np.shape(X)
    Y=np.zeros((8,M))
    for i in range(M):
        Y[0,i]=np.mean(X[:,i])
        Y[1,i]=np.var(X[:,i])
        Y[2,i]=np.sqrt(np.sum(X[:,i]**2))
        E=relative_energy(X[:,i],5,1)
        Y[3:8,i]=E
    return Y

def my_pca(X):
    D,M=np.shape(X)
    X_=np.zeros((D,M))
    for d in range(D):
        X_[d,:]=(X[d,:]-np.mean(X[d,:]))/np.std(X[d,:])
    U,S,Vt=np.linalg.svd(X_)
    var_exp=S**2/(M-1)
    S2=np.zeros((D,M))
    S2[:D,:D]=np.diag(S)
    return U, np.dot(S2,Vt), var_exp

def plot_correlation_circle(U,feature_names,var_exp):
    D,D=np.shape(U)
    figure, axes = plt.subplots()
    for d in range(D):
        plt.plot([0,U[d,0]],[0,U[d,1]])
        plt.text(U[d,0]+0.01, U[d,1]+0.01,feature_names[d])
    theta = np.linspace(0, 2*np.pi, 100)
    radius = 1
    a = radius*np.cos(theta)
    b = radius*np.sin(theta)
    axes.set_aspect(1)
    v=np.round(1000*var_exp/np.sum(var_exp))/10
    plt.plot(a,b,'k')   
    plt.xlabel("PC1 : "+str(v[0])+" % of total variance")
    plt.ylabel("PC2 : "+str(v[1])+" % of total variance")
    plt.show()


# === CONFIG ===
folder = "motif_csv/"
target_len = None       # set to e.g. 2000 if you want to force same length
n_components = 5        # number of PCA dimensions
k = 3                   # number of clusters
random_state = 42

# === Load all motifs ===
motifs = []
names = []

for file in sorted(os.listdir(folder)):
    file_path = os.path.join(folder, file)
    if not file.endswith(".csv"):
        continue
    path = os.path.join(folder, file)
    data = np.loadtxt(path, delimiter=",")

    motif = np.loadtxt(file_path, delimiter=",")
    Fs = 32000
    N = np.size(motif)
    t=np.arange(N)/Fs
    
    # optional: enforce same length (trim or pad)
    if target_len is not None:
        if len(data) > target_len:
            data = data[:target_len]
        elif len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)))
    motifs.append(data)
    names.append(file)

motifs = np.array(motifs)  # shape: (N_motifs, N_samples)
print(f"✅ Loaded {len(motifs)} motifs, each of length {motifs.shape[1]} samples")

Y1=compute_features(X1)
Y2=compute_features(X2)


 # Computation of PCA
Y =np.concatenate((Y1,Y2),axis=1)
D,M=np.shape(Y)
U,S_,var_exp=my_pca(Y)

feature_names = [
  'mean',
  'var',
  'RMSE',
  'E1', 'E2', 'E3', 'E4','E5']
plot_correlation_circle(U,feature_names,var_exp)







