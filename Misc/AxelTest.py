#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:21:55 2019

@author: axelnathanson
"""

import matplotlib
import sympy

np.set_printoptions(precision=2)

# sign of sqrt: u221A

eps = Symbol('\u03B5')
V = Symbol('V')
N = 9
j = N/2


H = zeros(N+1)


def Jz(n):
    m = j - n
    H[n,n]=-eps*m    

def J2(n):
    m = -j+n
    if n+2 <= N:
        H[n,n+2] = V/2 * sqrt((j-m)*(j+m+1)*(j-m-1)*(j+m+2))
    if n-2>=0:
        H[n,n-2] = V/2 * sqrt((j+m)*(j-m+1)*(j+m-1)*(j-m+2))
    



for x in range(0,N+1):
    Jz(x)
    J2(x)





print(H)
pprint(cse(H), use_unicode=True)


