import numpy as np
from math import *

def bestknown(dim,N):
    cor=(dim-1)*N/dim/(N-1)
    return sqrt(cor)

def genorth(Mt,M,Qpre):
    # Mt dimensionality of space
    # M dimensionality of subspaces
    # Qpre number of subspaces
    q = np.random.randn(Mt, M * Qpre)
    q = q.astype(np.complex64)
    q /= np.linalg.norm(q, axis=0)
    return q

def genweight(Mt,M,Qpre):
    WM = np.ones((Qpre, Qpre), np.complex64)
    return np.tril(WM, -1)
    
    WM = np.ones((Mt, M*Qpre), np.complex64)
    return np.tril(WM, -1)


    WM=np.zeros((Mt,M*Qpre), np.complex64)
    for i in range(Qpre):
        for j in range(i, Qpre):
            WM[i,j] = 1
    return 1-WM # np.tril(WM, -1)

print('======================================================')
print('This code generates complex Grassmannian Line Packings.')
print('======================================================')
Mt=int(input('Please enter the dimension of the complex space '))
M=1
skip=0
Qpre=int(input('Please enter the number of lines '))
QLeft=Qpre
correq=bestknown(Mt,Qpre)
print('The best distance in this case is ', correq)

def objectivefn(F):
    global M
    global Qpre
    global power
    global WM
    Mt=F.shape[0]
    Qpred=Qpre;
    cor=1-(Mt-1)*Qpred/Mt/(Qpred-1);
    if Qpre>Mt**2:
        cor=0
    fac=np.sqrt(cor);
    faclog=((1+fac * fac)*log(1+fac * fac))
    MM=F.conjugate().T @ F;

    if np.isnan(F).any(): raise
    if np.isnan(MM).any(): raise
    MM=(1+(MM * MM)) * (np.log(1+ MM * MM))
    MM=(MM-faclog)#* WM;
    ee2=MM.reshape(-1, 1)
    return (np.mean((ee2) ** power))

def objectivefn2(F):
    global M
    global Qpre
    global WM
    Mt=F.shape[0]
    Qpred=Qpre;
    cor=1-(Mt-1)*Qpred/Mt/(Qpred-1);
    if Qpre>Mt**2:
        cor=0
    fac=sqrt(cor);
    faclog=((1+fac*fac)*log(1+fac*fac));
    MM=F.conjugate().T @ F;
    MM=(1+(MM * MM)) * (np.log(1+MM * MM))
    MM=(MM-faclog) * WM
    ee=MM.reshape(-1, 1)
    return np.max(ee)

def orthcon(x,objectivefn,mag):
    global M, Mt
    global Qpre
    global skip
    maxiter=20;
    xbest=x;
    fbest=objectivefn(x);
    direction=np.concatenate([np.zeros((1,skip)), np.ones((1,(Qpre-skip)))], axis=1)
    direction=direction.astype(np.complex64)
    rate=0;
    for k in range(maxiter):
        gamma=np.ones((1,Qpre), np.complex64);
        Dx=mag*(NumGradI(objectivefn,x));
        Z=[]
        ttrace=[]
        cont=[]
        
        for jj in range(Qpre):
            f=x[:,(jj-1)*M+1:jj*M +1]
            h=f @ f.conjugate().T
            X=Dx[:,(jj-1)*M+1:jj*M+1]
            I1=np.eye(h.shape[0]);
            con1=(h-I1);
            z1=-(I1-h) @ (X)
            cont.append(con1)
            Z.append(z1)
            ttrace.append(np.trace(z1.conjugate().T @  z1))
            '''
            if cont is not None:
                cont=np.concatenate([cont, con1], axis=1)
            else: cont = con1
            if Z is not None:
                Z=np.concatenate([Z, z1], axis=1)
            else: Z = z1
            if ttrace is not None:
                ttrace=np.concatenate([ttrace, np.trace(z1.conjugate().T @  z1)], axis=1)
            else: ttrace = np.trace(z1.conjugate().T @  z1)
            '''
        Z = np.asarray(Z).squeeze().T
        ttrace = np.asarray(ttrace, np.complex64)
        cont = np.asarray(cont, np.complex64)
        #print(x.shape, ttrace.shape, Z.shape)
        XX=x+Z;
        xnn=np.copy(x);
        xn=np.copy(xnn);
        dd=objectivefn(x);
        for jj in range(skip,Qpre): #for jj in range(Qpre):
            f=x[:,(jj-1)*M+1:jj*M+1]
            h=f @ f.conjugate().T
            xn=np.copy(xnn);
            XX1=XX[:,(jj-1)*M+1:jj*M+1]
            
            Q1, R1=np.linalg.qr(XX1)
            xn[:,(jj-1)*M+1:jj*M]=Q1[:,1:M]
            QRfac=objectivefn(xn);
            ii=0;

            QRfacinit=QRfac;
            fact=1;
            
            while (dd-QRfac)>= 0 and ii<50:
                gamma[:,jj]=2*gamma[:,jj]
                temp=x[:,(jj-1)*M+1:jj*M+1] + gamma[:,jj] * Z[:,(jj-1)*M+1:jj*M+1]
                Q1, R1 = np.linalg.qr(temp);
                xn[:,(jj-1)*M+1:jj*M]=Q1[:,1:M]
                QRfac=objectivefn(xn)
                if QRfac>QRfacinit:
                    break
                QRfacinit=QRfac;
                ii=ii+1;
                fact=2;
            gamma[:,jj]=gamma[:,jj]/fact;
            temp=x[:,(jj-1)*M+1:jj*M+1]+gamma[:,jj]*Z[:,(jj-1)*M+1:jj*M+1]
            Q1, R1=np.linalg.qr(temp)
            xn[:,(jj-1)*M+1:jj*M]=Q1[:,1:M]
            QRfac=objectivefn(xn)
            ii=0;
            fact=1;
            while (dd-QRfac)<0 and ii<30:
                gamma[:,jj]=gamma[:,jj]*0.25;
                temp=x[:,(jj-1)*M+1:jj*M+1]+gamma[:,jj]*Z[:,(jj-1)*M+1:jj*M+1]
                Q1, R1 = np.linalg.qr(temp);
                xn[:,(jj-1)*M+1:jj*M+1]=Q1[:,1:M+1];
                QRfac=objectivefn(xn);
                ii=ii+1;

            gamma[:,jj]=gamma[:,jj]/fact;
            temp=x[:,(jj-1)*M+1:jj*M+1]+gamma[:,jj] * Z[:,(jj-1)*M+1:jj*M+1];
            Q1, R1 = np.linalg.qr(temp);
            xnn[:,(jj-1)*M+1:jj*M]=Q1[:,1:M];
            dd=objectivefn(xnn)
        xn=np.copy(x);
        for jj in range(skip, Qpre):
                XX1=xnn[:,(jj-1)*M+1:jj*M]
                Q1, R1=np.linalg.qr(XX1);
                xn[:,(jj-1)*M+1:jj*M]=Q1[:,1:M]
        x=np.copy(xn);
        fval=objectivefn(x)
        print('iteration number = ', k, '  Function value = ',fval)
        mm=fval
        if mm<fbest:
            print('fbest changed');
            fbest=mm;
            xbest=x;
            rate=0;

        elif mm>=fbest:
            rate=rate+1;
            if rate==2:
                x=xbest;
                mag=mag/10;
            if rate==4:
                x=xbest;
                mag=mag*100;

            if rate==6:
                x=xbest;
                mag=mag/1000;

            if rate==10:
                x=xbest
    return x, fval


def NumGradI(fcn,x):
    global M
    f0 = fcn(x)
    t=np.zeros(x.shape, np.complex64)
    delta = 1e-4;
    gradR=np.copy(t);
    gradI=np.copy(t);
    for ii in range(x.shape[0]):
        for kk in range(x.shape[1]):
            z = np.copy(t)

            z[ii,kk]=1*delta;
            xx=x+z
            f1=fcn(xx)
            gradR[ii,kk]=(f1-f0)/delta;
        
    for ii in range(x.shape[0]):
        for kk in range(x.shape[1]):
            z=np.copy(t)
            z[ii,kk]=1j*delta;
            xx=x+z;
            f1=fcn(xx)
            gradI[ii,kk]=(f1-f0)/delta;
        
    return gradR+1j*gradI;


x=genorth(Mt,M,QLeft)
WM=genweight(Mt,M,Qpre)
tol=10^-4
mag=1;
ma=10;
power=2;
for j in range(200):
    mag=mag*10
    x,fval=orthcon(x, objectivefn, mag)
    x,fval=orthcon(x,objectivefn2,ma)
    if fval<tol: break
    power=power+2
    if power==20:
        power=2
        mag=1
    if (j+1) % 5 == 0:
        ss=input('Do you wish to continue? Press 1 to stop or Enter to continue');
        if ss=='1':
            break

print('End of Optimzation ')
print('===================')
print(x)
print(1 - x.T @ x)
print('The best known distance in this case is ',correq)
