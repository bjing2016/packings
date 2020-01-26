import torch
import tqdm
from math import *
import numpy as np

def bestknown(dim,N):
    cor=(dim-1)*N/dim/(N-1)
    return sqrt(cor)

def orthorandom(d, k):
    m = torch.randn(d, k)
    m.div_(torch.norm(m, dim=0))
    return m

def normalize(m):
    with torch.no_grad():
        m.div_(torch.norm(m, dim=0))

def cosims(m):
    return m.T @ m

def loss(m):
    return torch.tril(cosims(m)**2, diagonal=-1)

def softmax(m, beta):
    return torch.sum(m ** (beta/2)) ** (1/beta)

def distro(loss):
    n = loss.shape[0]
    scale = (n-1)/2/n
    sqmean = loss.mean() / scale
    mean = loss.sqrt().mean() / scale
    return mean, (sqmean - mean**2).sqrt()

m = orthorandom(d=7, k=100).requires_grad_()
m = torch.from_numpy(np.load('k4n10.npy')).requires_grad_()

maxes = [1]

optim = torch.optim.SGD((m,), 0.01)
for i in tqdm.trange(210000):
    optim.zero_grad()
    myloss = loss(m)
    beta = 10 + 0.0005 * i
    minmax = softmax(myloss, beta)
    if i % 1000 == 0: 
        with torch.no_grad(): 
            mymax = torch.max(myloss).item()
            maxes.append(mymax)
            print(beta, mymax)
        tosave = m.detach().numpy()
        if min(maxes) == mymax:
            np.save('k7n100.npy', m.detach().numpy())
    minmax.backward()
    optim.step()
    normalize(m)

m = m.detach().numpy()

# np.arccos(abs(m.T@m)) * 180/np.pi