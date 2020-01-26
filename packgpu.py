import torch
import numpy as np
import tqdm

def orthorandom(d, k):
    m = torch.randn(d, k)
    m.div_(torch.norm(m, dim=0))
    return m.cuda()

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
    return mean, (sqmean - mean**2)


import glob
m = sorted(glob.glob('*.npy'))[0]
m = torch.from_numpy(np.load(m)).cuda()
#m += torch.rand_like(m) * 1e-10
m = m.requires_grad_()

#m = None
if m is None: m = orthorandom(d=256, k=1000).requires_grad_()
    
optim = torch.optim.SGD((m,), 0.01)
maxes = [float(sorted(glob.glob('*.npy'))[0][:-4])]
beta = 10
for i in tqdm.trange(100000):
    optim.zero_grad()
    myloss = loss(m)
    if i < 20000: beta += 0.001
    else: beta += 0.0001
    minmax = softmax(myloss, beta)
    if i % 1000 == 0: 
        with torch.no_grad(): 
            mymax = torch.max(myloss).item()
        print(beta, mymax)
        maxes.append(mymax)
        if min(maxes) == maxes[-1]:
            tm = m.detach().cpu().numpy()
            np.save(str(mymax), tm)
        
    minmax.backward()
    optim.step()
    normalize(m)

m = m.detach().cpu().numpy()
np.save(str(torch.max(myloss).item()), m)

