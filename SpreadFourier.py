#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:51:16 2020

@author: Kevin
"""

from scipy import interpolate
import scipy as sy
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm


def utest(u_bar, X0, N):        
    for i in range(N):
        u_test1 = np.pi * (i - N/2) / X0
        if (u_test1 > u_bar):
            return u_test1
    return u_test1       

def P_hat(u):
    value = sy.special.gamma(1j*(u[0]+u[1])-1) * sy.special.gamma(-1j*u[1]) / sy.special.gamma(1j*u[0]+1)
    return value
        
def Phi(u,r,sigma1,sigma2,rho,tau):
    rvec = np.array([[r], [r]])
    sigma = np.array([[sigma1**2], [sigma2**2]])
    Sigma = np.array([[sigma1**2, rho*sigma1*sigma2], [rho*sigma1*sigma2, sigma2**2]])
    value = np.exp(1j*np.matmul(u,(rvec - 0.5*sigma))*tau - np.matmul(np.matmul(u,Sigma),np.transpose(u))*tau*0.5);
    return value

class spread:
    def __init__(self, S1, S2, K, r, sigma1, sigma2, rho, tau, N):

        self.u_bar = 40;
        self.e1 = -3;
        self.e2 = 1;
                
        self.X1 = np.log(S1/K)
        self.X2 = np.log(S2/K)
        self.K = K
        self.r = r
        #self.d1 = d1
        #self.d2 = d2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.tau = tau
        self.N = N
    
        l = np.linspace(0,N-1,N)
            
        u_bar1 = utest(self.u_bar, self.X1, self.N)
        u_bar2 = utest(self.u_bar, self.X2, self.N)
            
        eta1 = 2*u_bar1 / N
        eta2 = 2*u_bar2 / N
        self.eta1 = eta1
        self.eta2 = eta2
        eta_star1 = np.pi / u_bar1
        eta_star2 = np.pi / u_bar2
        
        u1 = -u_bar1 + eta1*l
        u2 = -u_bar2 + eta2*l
        x1 = -0.5*N*eta_star1 + eta_star1*l
        x2 = -0.5*N*eta_star2 + eta_star2*l
        
        self.u1 = u1
        self.u2 = u2
        self.x1 = x1
        self.x2 = x2
        
        
            
        H = np.zeros(shape=(N,N), dtype=np.complex128)
        C = np.zeros(shape=(N,N), dtype=np.complex128)
        for i in range(N):
            for j in range(N):

                u = np.array([u1[i] + 1j * self.e1, u2[j] + 1j * self.e2])
                Phiv = Phi(u,r,sigma1,sigma2,rho,tau)
                P_hatv = P_hat(u)
                H[i,j] = pow(-1,i+j) * Phiv * P_hatv
                C[i,j] = pow(-1,i+j) * np.exp(-self.e1*x1[i]-self.e2*x2[j]) * eta1 * eta2 * (N/(2*np.pi))**2
        
        self.C = C
        self.H = H
        z1 = np.absolute(self.X1-x1)
        z2 = np.absolute(self.X2-x2)
        z1 = z1.tolist()
        z2 = z2.tolist()
        self.p1 = z1.index(min(z1));
        self.p2 = z2.index(min(z2));

    def Price(self):
        H = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        C = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        for i in range(self.N):
            for j in range(self.N):

                u = np.array([self.u1[i] + 1j * self.e1, self.u2[j] + 1j * self.e2])
                Phiv = Phi(u, self.r, self.sigma1, self.sigma2, self.rho, self.tau)
                P_hatv = P_hat(u)
                H[i,j] = pow(-1,i+j) * Phiv * P_hatv
                C[i,j] = pow(-1,i+j) * np.exp(-self.e1 * self.x1[i] - self.e2 * self.x2[j]) * self.eta1 * self.eta2 * (self.N/(2*np.pi))**2
                
        Vmat = self.K * np.exp(-self.r * self.tau) * np.real(C * np.fft.ifft2(H))
        print(Vmat[self.p1,self.p2])
        return self.x1, self.x2, Vmat
    
    def Delta(self):
        H1 = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        H2 = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        C = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        for i in range(self.N):
            for j in range(self.N):

                u = np.array([self.u1[i] + 1j * self.e1, self.u2[j] + 1j * self.e2])
                Phiv = Phi(u, self.r, self.sigma1, self.sigma2, self.rho, self.tau)
                P_hatv = P_hat(u)
                
                H1[i,j] = pow(-1,i+j) * (1j * self.u1[i] - self.e1) * Phiv * P_hatv
                H2[i,j] = pow(-1,i+j) * (1j * self.u2[j] - self.e2) * Phiv * P_hatv
                
                C[i,j] = pow(-1,i+j) * np.exp(-self.e1 * self.x1[i] - self.e2 * self.x2[j]) * self.eta1 * self.eta2 * (self.N/(2*np.pi))**2

        Delta1mat = np.exp(- self.X1) * np.exp(-self.r * self.tau) * np.real(C * np.fft.ifft2(H1))
        Delta2mat = np.exp(- self.X2) * np.exp(-self.r * self.tau) * np.real(C * np.fft.ifft2(H2))
        print(Delta1mat[self.p1,self.p2])
        print(Delta2mat[self.p1,self.p2])
        return self.x1, self.x2, Delta1mat, Delta2mat
    
    def Gamma(self):
        H1 = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        H2 = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        H11 = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        H12 = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        H22 = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        C = np.zeros(shape=(self.N, self.N), dtype=np.complex128)
        for i in range(self.N):
            for j in range(self.N):

                u = np.array([self.u1[i] + 1j * self.e1, self.u2[j] + 1j * self.e2])
                Phiv = Phi(u, self.r, self.sigma1, self.sigma2, self.rho, self.tau)
                P_hatv = P_hat(u)
                
                H1[i,j] = pow(-1,i+j) * (1j * self.u1[i] - self.e1) * Phiv * P_hatv
                H2[i,j] = pow(-1,i+j) * (1j * self.u2[j] - self.e2) * Phiv * P_hatv
                H11[i,j] = pow(-1,i+j) * (1j * self.u1[i] - self.e1)**2 * Phiv * P_hatv
                H12[i,j] = pow(-1,i+j) * (1j * self.u1[i] - self.e1) * (1j * self.u2[j] - self.e2) * Phiv * P_hatv
                H22[i,j] = pow(-1,i+j) * (1j * self.u2[j] - self.e2)**2 * Phiv * P_hatv
                    
                C[i,j] = pow(-1,i+j) * np.exp(-self.e1 * self.x1[i] - self.e2 * self.x2[j]) * self.eta1 * self.eta2 * (self.N/(2*np.pi))**2
                
        Gamma11mat = np.exp(- 2 * self.X1) / self.K * np.exp(-self.r * self.tau) * (-np.real(C * np.fft.ifft2(H1)) + np.real(C * np.fft.ifft2(H11)))
        Gamma12mat = np.exp(- self.X1 * self.X2) / self.K * np.exp(-self.r * self.tau) * np.real(C * np.fft.ifft2(H12))
        Gamma22mat = np.exp(- 2 * self.X2) / self.K * np.exp(-self.r * self.tau) * (-np.real(C * np.fft.ifft2(H2)) + np.real(C * np.fft.ifft2(H22)))
        print(Gamma11mat[self.p1,self.p2])
        return self.x1, self.x2, Gamma11mat, Gamma12mat, Gamma22mat

def spread_inter(S1, S2, K, r, sigma1, sigma2, rho, tau, N):
    
    SP = spread(S1, S2, K, r, sigma1, sigma2, rho, tau, N)
    
    x,y,v = SP.Price()
    
    xx, yy = np.meshgrid(x, y)
    
    f = interpolate.interp2d(x, y, v, kind='cubic')
    return f

def spread_inter1(S1, S2, K, r, sigma1, sigma2, rho, tau, N):
    
    SP = spread(S1, S2, K, r, sigma1, sigma2, rho, tau, N)
    
    x,y,d1,d2 = SP.Delta()
    #print(np.exp(x)*K)
    #print(np.exp(y)*K)
    
    xx, yy = np.meshgrid(x, y)
    
    f1 = interpolate.interp2d(x, y, d1, kind='quintic')
    f2 = interpolate.interp2d(x, y, d2, kind='quintic')
    return f1,f2

def spread_inter2(S1, S2, K, r, sigma1, sigma2, rho, tau, N):
    
    SP = spread(S1, S2, K, r, sigma1, sigma2, rho, tau, N)
    
    x,y,g11,g12,g22 = SP.Gamma()
    
    xx, yy = np.meshgrid(x, y)
    
    f11 = interpolate.interp2d(x, y, g11, kind='cubic')
    f12 = interpolate.interp2d(x, y, g12, kind='cubic')
    return f11,f12

'''
fig = plt.figure()
ax = fig.gca(projection='3d')
S1 = np.arange(1, 10, 1)
S2 = np.arange(1, 10, 1)
XX, YY = np.meshgrid(X, Y)

V = f(X,Y)

surf = ax.plot_surface(XX, YY, V, cmap=cm.coolwarm, linewidth=0, antialiased=False)
'''