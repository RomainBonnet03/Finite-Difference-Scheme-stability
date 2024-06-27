# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:28:48 2024

@author: dutym
"""

""" CREDIT:
This code uses the library "utils" implemented by LeBarbenchon:
Title:         boundaryscheme/utils.py
Author:        LeBarbenchon Pierre  
Date:          2023
Code Version:  2.3
Availabity:    https://github.com/PLeBarbenchon/boundaryscheme
The LICENCE allowing the use and modification of this code is provided in the
file to which is attached this file.   
"""

# Python library
import numpy as np

# LeBarbenchon library (cf Credit)
from Lebarbenchon_codes.utils import coefBinomial

class Boundary():
    def __init__(self):
        self.m        = 0             # initialization
        self.B = np.zeros((0,0))      # initialization
    
class Dirichlet(Boundary):
    def __call__(self,r):
         self.m = 1
         B_     = np.concatenate((np.eye(r),np.zeros((r,1))), axis=1)
         self.B = B_

class Neumann(Boundary):
    def __init__(self,kb):
        self.kb = kb

    def __call__(self, r):
        self.m = self.kb
        B_int = np.ones((r,self.m))
        
        def rec(j):
            Coeff_line = np.zeros((1,self.m))
            if j>=0:
                Coeff_line[0,j] = 1
            elif j == -1:
                Coeff_line[0,:] = np.array([(-1)**(k+1)*coefBinomial(self.m,k)\
                                            for k in range(1,self.m+1)])
            else:
                for k in range(1,self.m+1):
                    
                    Coeff_line += (-1)**(k+1)*coefBinomial(self.m,k)*rec(j+k)
            
            return Coeff_line
        
        for r_ in range(r):
            B_int[-r+r_,:] = rec(-r+r_)
        
        B_ = np.concatenate((np.eye(r),-B_int), axis=1)
        self.B = B_
