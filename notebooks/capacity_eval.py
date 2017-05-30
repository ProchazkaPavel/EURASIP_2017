import numpy as np
import time
import subprocess
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os, sys, inspect
import mpmath as mp
from compare_QPSK import const_design_XOR, QAM, const_superposed
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import linprog
#import _capacity_eval_lib

def eval_sigma2w(SNR):
  gammaAdB = float(SNR)
  gammaA= 10.0**(gammaAdB/10)
  return 1.0 / (gammaA)

def eval_sigma2w_gen(SNR, const):
  M = len(const) # Cardinality
  Es = float(np.sum(np.abs(const)**2)) / M # Energy per symbol (complex envelope)
  Eb = float(Es)/np.log2(M) # Energi per bit (complex envelope)
  gammaAdB = SNR
  gammaA= 10.0**(gammaAdB/10)
  return Eb / (gammaA)

def cap_eval_Gauss(SNR):
  sigma2w = eval_sigma2w(SNR)
  return np.log2(1 + 1./sigma2w)

def cap_eval_ef(SNR, const, M):
  sigma2w = eval_sigma2w(SNR)
  Md = len(const)
  px = np.zeros(Md * M, dtype=float)
  pxd = np.zeros(Md * M, dtype=float)
  d = np.zeros(Md * M, dtype=complex)
  for i in range (0, Md):
    d[(i * M):((i + 1) * M)] = np.repeat(const[i], M)
  w = np.sqrt(sigma2w/2) * (np.random.normal(size=M * Md) + 1j * np.random.normal(size = M * Md))
  x = d + w
  ex = np.zeros([Md, Md * M], dtype=float)
  for i in range (0, Md):
    ex[i,:] = 1.0 / (np.pi * sigma2w) * np.exp(-np.abs(x - const[i])**2/(sigma2w));
  px = ex.sum(axis=0) / Md
  for i in range (0, Md):
    pxd[(i * M):((i + 1) * M)] = ex[i, (i * M):((i + 1) * M)]
  Hx = - 1.0/M/Md * np.sum(np.log2(px)); 
  Hxd = - 1.0/M/Md * np.sum(np.log2(pxd));
#  print d,x, "\npx:", px,'\npdx:', pxd, '\n\nex:', ex  
#  print Hx, Hxd
  return Hx - Hxd; 
#  return px 

def eval_cap_HSI(Nb, Ns, SNR, M=1e5):
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, 1.) 
  if Nb > 0:
    return cap_eval_ef(SNR, basic_part, M)
  else:
    return 0.

def eval_cap_BC(Nb, Ns, SNR, M=1e5):
  return cap_eval_ef(SNR, QAM(Nb+2*Ns)[0], M)


def eval_MAC(Nb, Ns, SNR, M, h = 1): # XOR
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, 1.) 
  sup_A = const_superposed(Nb, Ns, 'XOR') # superposed part of A
  sub_B = sup_A * 1j  * h # superposed part of B
  bas_A = basic_part # basic part of A
  bas_B = basic_part * h # basic part of B
  sigma2w = eval_sigma2w(SNR) # Performance Related to gammaC
#  sigma2w = eval_sigma2w(SNR)
  Ms = len(sup_A)
  Mb = len(basic_part)
  d = np.zeros(M, dtype=complex)
  Hx = 0
  Hx_cAs = 0
  Hx_cBs = 0
  Hx_cb = 0
  Hx_cAsBs = 0
  Hx_cbAs = 0
  Hx_cbBs = 0
  Hx_cbAsBs = 0
  ex = np.zeros([(Ms*Mb)**2, M], dtype=float)
  for iAb in range (0, Mb):
    for iAs in range (0, Ms):
      for iBb in range (0, Mb):
        for iBs in range (0, Ms):
          d = bas_A[iAb] + bas_B[iBb] + sup_A[iAs] + sub_B[iBs]
	  w = np.sqrt(sigma2w/2) * (np.random.normal(size=(M)) + 1j * np.random.normal(size=(M))) # COMPLEX CASE
          x = np.zeros(M, dtype=complex)
          x = d + w
          px_cAs = np.zeros(M, dtype=float)
          px_cBs = np.zeros(M, dtype=float)
          px_cb = np.zeros(M, dtype=float)
          px_cAsBs = np.zeros(M, dtype=float)
          px_cbAs = np.zeros(M, dtype=float)
          px_cbBs = np.zeros(M, dtype=float)
          px_cbAsBs = np.zeros(M, dtype=float)
          for jAb in range (0, Mb):
            for jAs in range (0, Ms):
              for jBb in range (0, Mb):
                for jBs in range (0, Ms):
                  ind = jAb * Mb * Ms**2 + jAs * Mb * Ms + jBb * Ms + jBs
                  ex[ind,:] = 1.0 / (np.pi * sigma2w) * \
                     np.exp(-np.abs(x.flatten() - bas_A[jAb] -\
                     bas_B[jBb] - sup_A[jAs] - sub_B[jBs])**2/(sigma2w))

# Probabilities (conditional) evaluation
          for jAb in range (0, Mb):
            for jAs in range (0, Ms):
              for jBb in range (0, Mb):
                for jBs in range (0, Ms):
                  ind = jAb * Mb * Ms**2 + jAs * Mb * Ms + jBb * Ms + jBs;
	          px_cAs   += ex[ind,:] * (jAs == iAs) / Mb**2/ Ms; # p(x|As)
	          px_cBs   += ex[ind,:] * (jBs == iBs) / Mb**2/ Ms; # p(x|Bs)
	          px_cb    += ex[ind,:] * ((jAb^jBb) == (iAb^iBb)) / Mb/ Ms**2; # p(x|Ab+Bb)
	          px_cAsBs += ex[ind,:] * (jAs == iAs) * (jBs == iBs) / Mb**2; # p(x|As,Bs)
	          px_cbAs  += ex[ind,:] * (jAs == iAs) * ((jAb^jBb) == (iAb^iBb)) / Mb / Ms; # p(x|As,b)
	          px_cbBs  += ex[ind,:] * (jBs == iBs) * ((jAb^jBb) == (iAb^iBb)) / Mb / Ms; # p(x|Bs,b)
	          px_cbAsBs+= ex[ind,:] * (jAs == iAs) * (jBs == iBs) * ((jAb^jBb) == (iAb^iBb)) / Mb;  # p(x|Bs,b)

          px = ex.sum(axis=0) / Mb**2 / Ms**2 # p(x)
#          le = float(M)*4*Ms*Mb
          le = float(M)*Ms*Ms*Mb*Mb
          Hx        += -(np.sum(np.log2(px)) / le)
          Hx_cAs    += -(np.sum(np.log2(px_cAs)) / le)
          Hx_cBs    += -(np.sum(np.log2(px_cBs)) / le)
          Hx_cb     += -(np.sum(np.log2(px_cb)) / le)
          Hx_cAsBs  += -(np.sum(np.log2(px_cAsBs)) / le)
          Hx_cbAs   += -(np.sum(np.log2(px_cbAs)) / le)
          Hx_cbBs   += -(np.sum(np.log2(px_cbBs)) / le)
          Hx_cbAsBs += -(np.sum(np.log2(px_cbAsBs)) / le)
#  print Hx, Hx_cC, Hx_cAcB, Hx_cAcBcC, Hx_cAB, Hx_cABcC

  I3rd   = Hx       - Hx_cbAsBs
  I2ndAs = Hx_cAs   - Hx_cbAsBs
  I2ndBs = Hx_cBs   - Hx_cbAsBs
  I2ndb  = Hx_cb    - Hx_cbAsBs
  I1stAs = Hx_cbBs  - Hx_cbAsBs  
  I1stBs = Hx_cbAs  - Hx_cbAsBs
  I1stb  = Hx_cAsBs - Hx_cbAsBs
#  return [Hx, Hx_cC, Hx_cAcB, Hx_cAcBcC, Hx_cAB, Hx_cABcC]
  return np.array([I3rd,I2ndAs,I2ndBs,I2ndb,I1stAs,I1stBs,I1stb])

def eval_MAC_ef(Nb, Ns, SNR, M = 1e5, h=1.): # XOR - cython
  out = np.zeros(7)
  sigma2w = eval_sigma2w(SNR)
  sup = const_superposed(Nb, Ns, 'XOR')
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)
  #_capacity_eval_lib.eval_MAC_func(out, np.real(basic_part).copy(order='C'), \
  #np.imag(basic_part).copy(order='C'), np.real(basic_part * h).copy(order='C'), \
  #np.imag(basic_part * h).copy(order='C'), np.real(sup).copy(order='C'), \
  #np.real(sup * 1j * h).copy(order='C'), np.imag(sup * 1j * h).copy(order='C'), Nb, Ns, sigma2w, int(M), h)
  #return out

def eval_MAC_rand_phase(Nb, Ns, SNR, M1 = 100, M2 = 100):
  hAabs = np.random.rayleigh(1./np.sqrt(2),size = M1)
  hBabs = np.random.rayleigh(1./np.sqrt(2),size = M1)
  hA = hAabs * np.exp(1j * np.random.uniform(0,2*np.pi, size = M1))
  hB = hBabs * np.exp(1j * np.random.uniform(0,2*np.pi, size = M1))
  h_list = hA / hB 
#  for h in h_list:
#    cap += eval_MAC_ef(Nb, Ns, SNR, M2, h)
  cap = Parallel(n_jobs=8)(delayed(eval_MAC_ef)(Nb, Ns, SNR, M2, h)  for h in h_list)
  return np.asarray(cap).sum(axis=0) / float(M1)

def eval_MAC_test(Nb, Ns, SNR, M): # XOR
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, 1.) 
  sup_A = const_superposed(Nb, Ns, 'XOR') # superposed part of A
  sub_B = sup_A * 1j # superposed part of B
  bas_A = basic_part # basic part of A
  bas_B = basic_part # basic part of B
  np.savetxt('clib/sb_R.dat', np.real(bas_B), fmt='%.8f', delimiter=',')
  np.savetxt('clib/sb_I.dat', np.imag(bas_B), fmt='%.8f', delimiter=',')
  np.savetxt('clib/ss.dat', sup_A, fmt='%.8f', delimiter=',')
  sigma2w = eval_sigma2w(SNR) # Performance Related to gammaC
#  sigma2w = eval_sigma2w(SNR)
  Ms = len(sup_A)
  Mb = len(basic_part)
  d = np.zeros(M, dtype=complex)
  Hx = 0
  Hx_cAs = 0
  Hx_cBs = 0
  Hx_cb = 0
  Hx_cAsBs = 0
  Hx_cbAs = 0
  Hx_cbBs = 0
  Hx_cbAsBs = 0
  ex = np.zeros([(Ms*Mb)**2, M], dtype=float)
  for iAb in range (0, Mb):
    for iAs in range (0, Ms):
      for iBb in range (0, Mb):
        for iBs in range (0, Ms):
          d = bas_A[iAb] + bas_B[iBb] + sup_A[iAs] + sub_B[iBs]
	  w = np.sqrt(sigma2w/2) * (np.random.normal(size=(M)) + 1j * np.random.normal(size=(M))) # COMPLEX CASE
          x = np.zeros(M, dtype=complex)
          x = d + w
          px_cAs = np.zeros(M, dtype=float)
          px_cBs = np.zeros(M, dtype=float)
          px_cb = np.zeros(M, dtype=float)
          px_cAsBs = np.zeros(M, dtype=float)
          px_cbAs = np.zeros(M, dtype=float)
          px_cbBs = np.zeros(M, dtype=float)
          px_cbAsBs = np.zeros(M, dtype=float)
          for jAb in range (0, Mb):
            for jAs in range (0, Ms):
              for jBb in range (0, Mb):
                for jBs in range (0, Ms):
                  ind = jAb * Mb * Ms**2 + jAs * Mb * Ms + jBb * Ms + jBs
                  ex[ind,:] = 1.0 / (np.pi * sigma2w) * \
                     np.exp(-np.abs(x.flatten() - bas_A[jAb] -\
                     bas_B[jBb] - sup_A[jAs] - sub_B[jBs])**2/(sigma2w))

# Probabilities (conditional) evaluation
          for jAb in range (0, Mb):
            for jAs in range (0, Ms):
              for jBb in range (0, Mb):
                for jBs in range (0, Ms):
                  ind = jAb * Mb * Ms**2 + jAs * Mb * Ms + jBb * Ms + jBs;
	          px_cAs   += ex[ind,:] * (jAs == iAs) / Mb**2/ Ms; # p(x|As)
	          px_cBs   += ex[ind,:] * (jBs == iBs) / Mb**2/ Ms; # p(x|Bs)
	          px_cb    += ex[ind,:] * ((jAb^jBb) == (iAb^iBb)) / Mb/ Ms**2; # p(x|Ab+Bb)
	          px_cAsBs += ex[ind,:] * (jAs == iAs) * (jBs == iBs) / Mb**2; # p(x|As,Bs)
	          px_cbAs  += ex[ind,:] * (jAs == iAs) * ((jAb^jBb) == (iAb^iBb)) / Mb / Ms; # p(x|As,b)
	          px_cbBs  += ex[ind,:] * (jBs == iBs) * ((jAb^jBb) == (iAb^iBb)) / Mb / Ms; # p(x|Bs,b)
	          px_cbAsBs+= ex[ind,:] * (jAs == iAs) * (jBs == iBs) * ((jAb^jBb) == (iAb^iBb)) / Mb;  # p(x|Bs,b)

          px = ex.sum(axis=0) / Mb**2 / Ms**2 # p(x)
#          le = float(M)*4*Ms*Mb
          le = float(M)*Ms*Ms*Mb*Mb
          Hx        += -(np.sum(np.log2(px)) / le)
          Hx_cAs    += -(np.sum(np.log2(px_cAs)) / le)
          Hx_cBs    += -(np.sum(np.log2(px_cBs)) / le)
          Hx_cb     += -(np.sum(np.log2(px_cb)) / le)
          Hx_cAsBs  += -(np.sum(np.log2(px_cAsBs)) / le)
          Hx_cbAs   += -(np.sum(np.log2(px_cbAs)) / le)
          Hx_cbBs   += -(np.sum(np.log2(px_cbBs)) / le)
          Hx_cbAsBs += -(np.sum(np.log2(px_cbAsBs)) / le)
  print Hx, Hx_cAs, Hx_cBs, Hx_cb, Hx_cAsBs, Hx_cbAs, Hx_cbBs, Hx_cbAsBs 

  I3rd   = Hx       - Hx_cbAsBs
  I2ndAs = Hx_cAs   - Hx_cbAsBs
  I2ndBs = Hx_cBs   - Hx_cbAsBs
  I2ndb  = Hx_cb    - Hx_cbAsBs
  I1stAs = Hx_cbBs  - Hx_cbAsBs  
  I1stBs = Hx_cbAs  - Hx_cbAsBs
  I1stb  = Hx_cAsBs - Hx_cbAsBs
#  return [Hx, Hx_cC, Hx_cAcB, Hx_cAcBcC, Hx_cAB, Hx_cABcC]
  return np.array([I3rd,I2ndAs,I2ndBs,I2ndb,I1stAs,I1stBs,I1stb])

# Capacity in destination assuming error-less MAC - Related to Dest A
def eval_cap_overall(Nb, Ns, SNR_HSI, SNR_BC, M, hD = 0.2): # 
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, 1.) 
  sup_B = const_superposed(Nb, Ns, 'XOR') * 1j # superposed part of A
  sup_A = sup_B * 1j * hD # superposed part of B
  bas_A = basic_part * hD # basic part of A
  bas_B = basic_part # basic part of B
  sigma2w1 = eval_sigma2w(SNR_HSI) 
  sigma2w2 = eval_sigma2w(SNR_BC) 
#  sigma2w = eval_sigma2w(SNR)
  Ms = len(sup_A)
  Mb = len(basic_part)
  d = np.zeros(M, dtype=complex)
  conBC = QAM(Nb + 2*Ns)[0]
  Hyz = 0
  Hyz_cA = 0
  Hyz_cAs = 0
  Hyz_cAb = 0
  eyz = np.zeros([(Ms*Mb)**2, M], dtype=float)
  norm = 1.0 / (np.pi * sigma2w1) * 1.0 / (np.pi * sigma2w2)
  le = float(M)*Ms*Ms*Mb*Mb
  for iAb in range (0, Mb):
    for iAs in range (0, Ms):
      for iBb in range (0, Mb):
        for iBs in range (0, Ms):
          indBC = Mb * Ms * iAs + Mb * iBs + (iAb ^ iBb) 
#          indBC = 8 * ((iAb ^ iBb) >> 1) + iAs +  4 * iBs + 2 * ((iAb ^ iBb) & 1)
          d1 = bas_A[iAb] + sup_A[iAs] +  bas_B[iBb] + sup_B[iBs] # data for the first time slot
          d2 = conBC[indBC]
	  w1 = np.sqrt(sigma2w1/2) * (np.random.normal(size=(M)) + 1j * np.random.normal(size=(M))) # COMPLEX CASE
	  w2 = np.sqrt(sigma2w2/2) * (np.random.normal(size=(M)) + 1j * np.random.normal(size=(M))) # COMPLEX CASE
          y = np.zeros(M, dtype=complex)
          y = d1 + w1
          z = np.zeros(M, dtype=complex)
          z = d2 + w2
          pyz = np.zeros(M, dtype=float)
          pyz_cA = np.zeros(M, dtype=float)
          pyz_cAs = np.zeros(M, dtype=float)
          pyz_cAb = np.zeros(M, dtype=float)
          for jAb in range (0, Mb):
            for jAs in range (0, Ms):
              for jBb in range (0, Mb):
                for jBs in range (0, Ms):
                  ind = jAb * Mb * Ms**2 + jAs * Mb * Ms + jBb * Ms + jBs
                  indBC = Mb * Ms * jAs + Mb * jBs + (jAb ^ jBb)
#                  indBC = 8 * ((iAb ^ iBb) >> 1) + iAs +  4 * iBs + 2 * ((iAb ^ iBb) & 1)
                  eyz[ind,:] = norm * np.exp(-np.abs(y.flatten() - bas_A[jAb] - sup_A[jAs] - \
                                             bas_B[jBb] - sup_B[jBs])**2/(sigma2w1)) * \
                                      np.exp(-np.abs(z.flatten() - conBC[indBC])**2/(sigma2w2))

# Probabilities (conditional) evaluation
          for jAb in range (0, Mb):
            for jAs in range (0, Ms):
              for jBb in range (0, Mb):
                for jBs in range (0, Ms):
                  ind = jAb * Mb * Ms**2 + jAs * Mb * Ms + jBb * Ms + jBs;
	          pyz_cA   += eyz[ind,:] * (jAs == iAs) * (jAb == iAb)  / Mb    / Ms # p(y,z|a)
	          pyz_cAs  += eyz[ind,:] * (jAs == iAs)                 / Mb**2 / Ms # p(y,z|sA_s)
	          pyz_cAb  += eyz[ind,:] * (jAb == iAb)                 / Ms**2 / Mb     # p(y,z|sB_b)

          pyz = eyz.sum(axis=0) / Mb**2 / Ms**2 # p(x)
          Hyz        += -(np.sum(np.log2(pyz)) / le)
          Hyz_cA     += -(np.sum(np.log2(pyz_cA)) / le)
          Hyz_cAs    += -(np.sum(np.log2(pyz_cAs)) / le)
          Hyz_cAb    += -(np.sum(np.log2(pyz_cAb)) / le)
#  print Hx, Hx_cC, Hx_cAcB, Hx_cAcBcC, Hx_cAB, Hx_cABcC

  I  = Hyz     - Hyz_cA
  Ib = Hyz_cAs - Hyz_cA
  Is = Hyz_cAb - Hyz_cA
  return np.array([I, Ib, Is])

def eval_overall_rand_phase(Nb, Ns, SNR_HSI, SNR_BC, M1 = 300, M2 = 300, alpha = 0.8):
  hAabs = np.random.rayleigh(1./np.sqrt(2),size = M1)
  hBabs = np.random.rayleigh(1./np.sqrt(2),size = M1)
  hA = hAabs * np.exp(1j * np.random.uniform(0,2*np.pi, size = M1))
  hB = hBabs * np.exp(1j * np.random.uniform(0,2*np.pi, size = M1))
  h_list = alpha * hA / hB 
  cap = Parallel(n_jobs=8)(delayed(eval_cap_overall)(Nb, Ns, SNR_HSI, SNR_BC, M2, h)  for h in h_list)
#  return np.asarray(cap).sum() / float(M1)
  return np.asarray(cap).sum(axis = 0) / float(M1)


def eval_HSI_gen(Nb, Ns, SNR_HSI, M, hD = 0.2): # HSI assuming errorless both BC and MAC
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, 1.) 
  bas_A = basic_part * hD# basic part of A
  bas_B = basic_part # basic part of B
  sigma2w = eval_sigma2w(SNR_HSI) 
  Mb = len(basic_part)
  d = np.zeros(M, dtype=complex)
  Hx = 0
  Hx_cAB = 0
  Hx_cAcB = 0
  ex = np.zeros([Mb**2, M], dtype=float)
  norm = 1.0 / (np.pi * sigma2w) 
  le = float(M)*Mb*Mb
  for iAb in range (0, Mb):
    for iBb in range (0, Mb):
      d = bas_A[iAb] +  bas_B[iBb]
      w = np.sqrt(sigma2w/2) * (np.random.normal(size=(M)) + 1j * np.random.normal(size=(M))) # COMPLEX CASE
      x = np.zeros(M, dtype=complex)
      x = d + w
      px = np.zeros(M, dtype=float)
      px_cAB = np.zeros(M, dtype=float)
      px_cAcB = np.zeros(M, dtype=float)
      for jAb in range (0, Mb):
        for jBb in range (0, Mb):
          ind = jAb * Mb + jBb 
          ex[ind,:] = norm * np.exp(-np.abs(x.flatten() - bas_A[jAb] - bas_B[jBb])**2/(sigma2w)) 

# Probabilities (conditional) evaluation
      for jAb in range (0, Mb):
        for jBb in range (0, Mb):
          ind = jAb * Mb + jBb 
	  px_cAB   += ex[ind,:] * ((jAb ^ jBb) == (iAb ^ iBb))  / Mb # p()
	  px_cAcB  += ex[ind,:] * (jAb == iAb) * (jBb == iBb) # p()

      px = ex.sum(axis=0) / Mb**2  # p(x)
      Hx          += -(np.sum(np.log2(px)) / le)
      Hx_cAB      += -(np.sum(np.log2(px_cAB)) / le)
      Hx_cAcB     += -(np.sum(np.log2(px_cAcB)) / le)
#  print Hx, Hx_cC, Hx_cAcB, Hx_cAcBcC, Hx_cAB, Hx_cABcC

  I  = Hx_cAB   - Hx_cAcB
  return I

def eval_HSI_rand_phase(Nb, Ns, SNR_HSI, M1 = 300, M2 = 300, alpha = 0.8):
  hAabs = np.random.rayleigh(1./np.sqrt(2),size = M1)
  hBabs = np.random.rayleigh(1./np.sqrt(2),size = M1)
  hA = hAabs * np.exp(1j * np.random.uniform(0,2*np.pi, size = M1))
  hB = hBabs * np.exp(1j * np.random.uniform(0,2*np.pi, size = M1))
  h_list = alpha * hA / hB 
  cap = Parallel(n_jobs=7)(delayed(eval_HSI_gen)(Nb, Ns, SNR_HSI, M2, h)  for h in h_list)
  return np.asarray(cap).sum() / float(M1)


## Evaluation of adaptive regions

def Eval_capacities(N, SNR_HSI, SNR_MAC, SNR_BC):
  # All possible tupples
  tuples = np.asarray([(i,j) for i in range(0, N+1) for j in range(0, N-i+1) if i+2*j<=N])[1:]
  ll = len(tuples)
  print 'Evaluation for\n', tuples
  cap_BC = []
  cap_HSI = []
  cap_MAC = []
  for i in range(ll):
    (Nb, Ns) = tuples[i]
    print 'evaluating capacity for the pair', (Nb, Ns)
    cap_BC.append(Parallel(n_jobs=8)(delayed(eval_cap_BC)(Nb, Ns, SNR, M=1e5) for  SNR in SNR_BC))
    print 'BC completed'
    cap_HSI.append(Parallel(n_jobs=8)(delayed(eval_cap_HSI)(Nb, Ns, SNR, M=1e5) for  SNR in SNR_HSI))
    print 'HSI completed'
    cap_MAC.append(Parallel(n_jobs=8)(delayed(eval_MAC_ef)(Nb, Ns, SNR, M=1e4) for  SNR in SNR_MAC))
    print 'MAC completed'
    
#    cap_BC.append(np.asarray([eval_cap_BC(Nb, Ns, SNR, M=1e5) for SNR in SNR_BC]))
#    cap_HSI.append(np.asarray([eval_cap_HSI(Nb, Ns, SNR, M=1e5) for SNR in SNR_HSI]))
#    cap_MAC.append(np.asarray([eval_MAC(Nb, Ns, SNR, M=1e5) for SNR in SNR_MAC]))
  f = open('Capacities.dat','w')
  pickle.dump([np.asarray(cap_BC), np.asarray(cap_HSI), np.asarray(cap_MAC)], f)
  f.close()

def Eval_capacities_rand_H(N, SNR_HSI, SNR_MAC, SNR_BC): # avarage capacities with random phase and DL
  # All possible tupples
  tuples = np.asarray([(i,j) for i in range(0, N+1) for j in range(0, N-i+1) if i+2*j<=N])[1:]
  ll = len(tuples)
  print 'Evaluation for\n', tuples
  cap_BC = []
  cap_HSI = []
  cap_MAC = []
  for i in range(ll):
    (Nb, Ns) = tuples[i]
    print 'evaluating capacity for the pair', (Nb, Ns)
    cap_BC.append(Parallel(n_jobs=7)(delayed(eval_cap_BC)(Nb, Ns, SNR, M=1e5) for  SNR in SNR_BC))
    print 'BC completed'
    cap_HSI.append([eval_HSI_rand_phase(Nb, Ns, SNR, M1 = 200, M2 = 200, alpha = 0.8) for SNR in SNR_HSI])
    print 'HSI completed'
    cap_MAC.append([eval_MAC_rand_phase(Nb, Ns, SNR, M1 = 200, M2 = 200) for  SNR in SNR_MAC])
    print 'MAC completed'
    
#    cap_BC.append(np.asarray([eval_cap_BC(Nb, Ns, SNR, M=1e5) for SNR in SNR_BC]))
#    cap_HSI.append(np.asarray([eval_cap_HSI(Nb, Ns, SNR, M=1e5) for SNR in SNR_HSI]))
#    cap_MAC.append(np.asarray([eval_MAC(Nb, Ns, SNR, M=1e5) for SNR in SNR_MAC]))
  f = open('Capacities_rand_phase.dat','w')
  res = [np.asarray(cap_BC), np.asarray(cap_HSI), np.asarray(cap_MAC)]
  pickle.dump(res, f)
  f.close()
  return res


def cap_inner(i, cap_MAC, cap_BC, cap_HSI, N,  SNR_MAC, SNR_BC, SNR_HSI):
  print i
  gMAC = SNR_MAC[i]
  c = [-1, -1]
  A = [[2, 1], [1, 1]]
  though = np.zeros(N)
  Throughput = np.zeros([len(SNR_BC), len(SNR_HSI)])
  act_map = np.zeros([len(SNR_BC), len(SNR_HSI)], int)
  for j in range(len(SNR_BC)):
    gBC = SNR_BC[j]
    for k in range(len(SNR_HSI)):
      gHSI = SNR_HSI[k]
      for l in range(N):
        r3 = cap_MAC[l,i,0]  
        r2As = cap_MAC[l,i,1]  
        r2Bs = cap_MAC[l,i,2]  
        r2b = cap_MAC[l,i,3]  
        r1As = cap_MAC[l,i,4]  
        r1Bs = cap_MAC[l,i,5]  
        r1b = cap_MAC[l,i,6]  
  
        Rbs = np.array([r2As,r2Bs]).min()
        Rbss = np.array([r3, cap_BC[l,j]]).min()
        Rs = np.array([r1As, r1Bs]).min()
        Rb = np.array([r1b, cap_HSI[l,k]]).min()
  
        x0_bounds = (0, Rs)
        x1_bounds = (0, Rb)
        b = [Rbss, Rbs]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds))
        though[l] = -res.items()[3][1]
      Throughput[j,k] = though.max()
      act_map[j,k] = though.argmax()
  return Throughput, act_map 

def cap_inner_rb_rs(i, cap_MAC, cap_BC, cap_HSI, N,  SNR_MAC, SNR_BC, SNR_HSI):
#  print i
  gMAC = SNR_MAC[i]
  c = [-1, -1]
  A = [[2, 1], [1, 1]]
  though = np.zeros(N)
  Throughput = np.zeros([len(SNR_BC), len(SNR_HSI), N, 2])
  for j in range(len(SNR_BC)):
    gBC = SNR_BC[j]
    for k in range(len(SNR_HSI)):
      gHSI = SNR_HSI[k]
      for l in range(N):
        r3 = cap_MAC[l,i,0]  
        r2As = cap_MAC[l,i,1]  
        r2Bs = cap_MAC[l,i,2]  
        r2b = cap_MAC[l,i,3]  
        r1As = cap_MAC[l,i,4]  
        r1Bs = cap_MAC[l,i,5]  
        r1b = cap_MAC[l,i,6]  
  
        Rbs = np.array([r2As,r2Bs]).min()
        Rbss = np.array([r3, cap_BC[l,j]]).min()
        Rs = np.array([r1As, r1Bs]).min()
        Rb = np.array([r1b, cap_HSI[l,k]]).min()
  
        x0_bounds = (0, Rs)
        x1_bounds = (0, Rb)
        b = [Rbss, Rbs]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds))
        though[l] = -res.items()[3][1]
        Throughput[j,k,l, :] = res.items()[4][1]
  return Throughput

def eval_rates_rb_rs(N, SNR_MAC, SNR_BC, SNR_HSI, tuples):
  [cap_BC, cap_HSI, cap_MAC] = pickle.load(open('Capacities2.dat','r'))
#  [cap_BC, cap_HSI, cap_MAC] = pickle.load(open('Capacities_rand_phase.dat','r'))  
  N = len(tuples)
  res = (Parallel(n_jobs=8)(delayed(cap_inner_rb_rs)\
                          (i, cap_MAC[:,:,:], cap_BC[:,:], cap_HSI[:,:], N, SNR_MAC, SNR_BC, SNR_HSI)\
                          for i in range(len(SNR_MAC))))
  temp = np.asarray(res)
#  Rb_Rs = temp[0,:,0,:,:]
  Rb_Rs = temp
    
  return Rb_Rs

def eval_rates(N, SNR_MAC, SNR_BC, SNR_HSI, tuples):
  [cap_BC, cap_HSI, cap_MAC] = pickle.load(open('Capacities.dat','r'))
#  [cap_BC, cap_HSI, cap_MAC] = pickle.load(open('Capacities_rand_phase.dat','r'))

  
  N = len(tuples)
  rB = np.zeros([len(SNR_MAC), len(SNR_BC), len(SNR_HSI), len(tuples)], float)
  rS = np.zeros([len(SNR_MAC), len(SNR_BC), len(SNR_HSI), len(tuples)], float)
  though = np.zeros(N)
  c = [-1, -1]
  A = [[2, 1], [1, 1], [1,0], [0,1]]
  x0_bounds = (0, None)
  x1_bounds = (0, None)
  par = 1
  if par == 0:
    for i in range(len(SNR_MAC)):
      print i
      gMAC = SNR_MAC[i]
      for j in range(len(SNR_BC)):
        gBC = SNR_BC[j]
        for k in range(len(SNR_HSI)):
          gHSI = SNR_HSI[k]
          for l in range(N):
            r3 = cap_MAC[l,i,0]  
            r2As = cap_MAC[l,i,1]  
            r2Bs = cap_MAC[l,i,2]  
            r2b = cap_MAC[l,i,3]  
            r1As = cap_MAC[l,i,4]  
            r1Bs = cap_MAC[l,i,5]  
            r1b = cap_MAC[l,i,6]  
  
            Rbs = np.array([r2As,r2Bs]).min()
            Rbss = np.array([r3, cap_BC[l,j]]).min()
            Rs = np.array([r1As, r1Bs]).min()
            Rb = np.array([r1b, cap_HSI[l,k]]).min()
  
            b = [Rbss, Rbs, Rs, Rb]
            res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds))
            though[l] = -res.items()[3][1]
          Throughput[i,j,k] = though.max()
          act_map[i,j,k] = though.argmax()
  else:
    res = []
    res.append(Parallel(n_jobs=8)(delayed(cap_inner)\
                          (i, cap_MAC, cap_BC, cap_HSI, N, SNR_MAC, SNR_BC, SNR_HSI)\
                          for i in range(len(SNR_MAC))))
    temp = np.asarray(res)
    Throughput = temp[0,:,0,:,:]
    act_map = temp[0,:,1,:,:]
    
  return Throughput, act_map

def draw_solution(Rbs, Rbss, Rs, Rb):
  c = [-1, -1]
  A = [[2, 1], [1, 1], [1,0], [0,1]]
  b = [Rbss, Rbs, Rs, Rb]
  x0_bounds = (0, None)
  x1_bounds = (0, None)
#  res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds), options={"disp": True})
  res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds))
  plt.figure()
  plt.plot([0,Rbss/2.],[Rbss, 0],'k-', lw=2, label='Third Order')
  plt.plot([0,Rbs],[Rbs, 0],'r-', lw=2, label='Active Sencond Order')
  plt.plot([Rs, Rs],[0, Rbss],'b-', lw=2, label='First Order')
  plt.plot([0,Rbss],[Rb, Rb],'b-', lw=2)
  plt.plot([res.items()[4][1][0]],[res.items()[4][1][1]],'co', ms=15)
  plt.legend()
  plt.show()

def test_through():
  SNR_MAC = np.linspace(-5,20,500)
  SNR_BC = np.linspace(-5,20,500)
  SNR_HSI = np.linspace(-5,20,500)
  N = 4
  Eval_capacities(N, SNR_HSI, SNR_MAC, SNR_BC)
#  print 'Evaluated cap'
#  tuples = np.asarray([(i,j) for i in range(0, N+1) for j in range(0, N-i+1) if i+2*j<=N])[1:]
#  Throughput, act_map = eval_rates(N, SNR_MAC, SNR_BC, SNR_HSI, tuples)
#  pickle.dump([Throughput, act_map], open('Region_results.dat','w'))
 
def Eval_Capacities_QAMs(SNR_range, N, M=1e5):
  cap = []
  for i in range(N):
    print 'Working on', str(i)
    const = QAM(i)[0]
    cap.append(Parallel(n_jobs=8)(delayed(cap_eval_ef)(SNR, const, int(M)) for  SNR in SNR_range))
  return cap

if __name__ == '__main__':
  SNR_MAC = np.linspace(-5,20,250)
  SNR_BC = np.linspace(-5,20,250)
#  SNR_BC = np.array([11])
  SNR_HSI = np.linspace(-5,20,250)
#  N = 4
#  res = Eval_capacities_rand_H(N, SNR_HSI, SNR_MAC, SNR_BC)
#  Eval_capacities(N, SNR_HSI, SNR_MAC, SNR_BC)
#  test_through()
#  tuples = np.asarray([(i,j) for i in range(0, N+1) for j in range(0, N-i+1) if i+2*j<=N])[1:]
#  Throughput, act_map = eval_rates(N, SNR_MAC, SNR_BC, SNR_HSI, tuples)
#  pickle.dump([Throughput, act_map], open('Region_results_rand_phase.dat','w'))
