from packgpu import *
import numpy as np

tasks = [
    #(32, 75),
    #(32, 100),
    #(32, 150),
    #(64, 150),
    #(64, 200),
    #(64, 250),
    #(64, 300),
    #(64, 400),
    #(64, 500),
    #(128, 500),
    #(128, 600),
    #(128, 700),
    #(128, 750),
    #(128, 800),
    #(128, 900),
    #(128, 1000),
    #(256, 1000),
    #(256, 1500),
    #(256, 2000),
    #(512, 2000),
    #(512, 3000),
    #(512, 4000),
    (512, 5000)
]

for d, k in tasks:
    m = orthorandom(d=d, k=k).requires_grad_()
    best, betas, maxes = train(m, schedule)    
    np.save('autobest/{}.{}best.npy'.format(d, k), best)
    np.save('autobest/{}.{}betas.npy'.format(d, k), betas)
    np.save('autobest/{}.{}maxes.npy'.format(d, k), maxes)

