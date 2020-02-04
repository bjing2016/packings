from packgpu import *
import numpy as np

tasks = [
    (64, 500),
    (128, 800),
    (128, 900),
]

for d, k in tasks:
    m = orthorandom(d=d, k=k)
    best, betas, maxes = train(m, schedule)    
    np.save('autobest/{}.{}best.npy'.format(d, k), best)
    np.save('autobest/{}.{}betas.npy'.format(d, k), betas)
    np.save('autobest/{}.{}maxes.npy'.format(d, k), maxes)

