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

def beta_cycle(optim, m, beta0, dbeta, betas, maxes):
    bestmax = 1
    best = None
    kmax = []
    i=0
    while True:
        optim.zero_grad()
        myloss = loss(m)
        beta = beta0 + i * dbeta
        minmax = softmax(myloss, beta)
        with torch.no_grad(): 
            mymax = torch.max(myloss).item()
        
        betas.append(beta)
        maxes.append(mymax)
        if mymax < bestmax:
            bestmax = mymax
            best = m.clone()
        minmax.backward()
        optim.step()
        normalize(m)
        i+=1
        if i % 1000 == 0:
            kmax.append(mymax)
            print(beta, mymax)      
            if len(kmax) > 6 and (kmax[-3] > kmax[-2] and kmax[-2] < kmax[-1]) or np.isnan(kmax[-1]):
                print('Next beta cycle')
                break
                    
    return bestmax, best

schedule = [
    (2, 10, 1e-3),
    (3, 10, 3e-4),
    (10, 20, 1e-4)
]

def train(m, beta_schedule):
    bestmax = 1
    best = m
    
    betas = []
    maxes = []
    
    for beta0, niters, dbeta in schedule:
        for i in range(niters):
            best = best.requires_grad_()
            optim = torch.optim.SGD((best,), 0.01)
            mymax, mybest = beta_cycle(optim, best, beta0, dbeta, betas, maxes)
            if mymax < bestmax:
                bestmax = mymax
                best = mybest.detach()
            else: 
                print('Next beta schedule')
                break
    return best.detach().cpu().numpy(), betas, maxes
