import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import erf
from scipy.special import erfc

from analytical_BER import *
from Numerical_BER import *

def dec2bin(x, num_bits=None): # Convert decimal number to binary narray
  return np.asarray([int(c)  for c in np.binary_repr(x,width=num_bits)],dtype=int) # dec to bin array

def bin2dec(vec): # Convert binary narray to decimal number
  return int(np.array_str(vec).strip('[]').replace(' ',''),2)

def revbinorder(x, num_bits=None):
  return bin2dec(dec2bin(x, num_bits)[-1::-1])

def index2vec(index, bits): # Convert a single index to multidimensional vector acording to sizes
  vec = np.zeros(len(bits), dtype=int)
  bin_vec = dec2bin(index, np.sum(bits))
#  print bin_vec
  start = 0
  for i in range(0, len(bits)):
    vec[i] = bin2dec(bin_vec[start:(start+bits[i])])
    start += bits[i]
  return vec



### Pairwise error approximation

# Approximate the BER using the most probable pairwise error and mean number of neights
def pair_wise_err(K, sigma2w, rho2):
  return  K * erfc(np.sqrt(float(rho2)/sigma2w)/2) / 2

# Evaluate BER for given NON-OVERLAPPING constellations
def eval_pair_wise_err(SNR, const):
  sigma2w = SNR2sigma2w(SNR)
  dist = np.round(np.asarray([np.abs(const[i] - const[j])**2 \
         for i in np.arange(0,len(const)) for j in np.arange(i+1,len(const))]),8)
  rho2 = dist.min()
  K = len(np.nonzero(dist == dist.min())[0]) * 2. / len(const) 
  return pair_wise_err(K, sigma2w, rho2)

# evaluate average number of neighboors and minimal distance for a given constellation
def eval_num_neigh(const, alpha = 0): # 
#in case of QAM based grid constellations, alpha is distance in grid 
#  Evaluation of avaradge number of nearest neighs
  if ((len(const) >= 2**12) and (alpha != 0)): # k -> 4 for large constellations
    return 4, 4 * alpha**2
  else:
    constr = np.sort_complex(const)
    constr_un = np.unique(np.round(constr,10)) # Unique constellation
    dist = np.round(np.asarray([np.abs(constr_un[i] - constr_un[j])**2\
           for i in np.arange(0,len(constr_un)) for j in np.arange(0,len(constr_un)) if (i != j)]),10)
    # Evaluation average number of nearest neighbours
    re_vals = np.unique(np.real(constr_un))
    im_vals = np.unique(np.imag(constr_un))
    SI_c = np.asarray([(np.real(i), np.imag(i)) for i in constr])
    hist = np.asarray([[((np.round(SI_c,10)[:,0] == re_vals[i]) * (np.round(SI_c,10)[:,1] == im_vals[j])).sum() \
           for i in range(0,len(re_vals))] for j in range(0,len(im_vals))])
    vec = np.nonzero(np.abs(dist.reshape(len(constr_un),len(constr_un) - 1) - dist.min()) < 1e-6)[0]
    vec2 = np.nonzero(np.diff(vec))[0]
    tt = np.hstack([vec2[0]+1, np.diff(vec2), len(np.diff(vec)) - vec2[-1]])
#    print 'AAA',const, tt*hist.flatten(), (tt*hist.flatten()).sum()/float(len(const))
    return (tt*hist.flatten()).sum()/float(len(const)), dist.min()


# evaluate average number of neighboors and minimal distance for a given constellation
def eval_num_neigh_gen(Nb, Ns, mapping = 'XOR', h =1): #
# const -> comlex vector
# mapping = b <-> index corresponds to symbol b
#in case of QAM based grid constellations, alpha is distance in grid 
#  Evaluation of avarage number of nearest neighs
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)
  if (len(relay_const) >= 2**12): # k -> 4 for large constellations
    print 'Warning, large constellation. Evaluation will take a while'
  dist_full = np.asarray([[np.abs(relay_const[i] - relay_const[j])**2\
         for i in np.arange(0,len(relay_const))] for j in np.arange(0,len(relay_const))])
  dists = []
  for i in range(len(relay_const)):
    iA = i >> (Nb+Ns) 
    iB = i % (2**(Nb+Ns))
    iAb = iA % (2**Nb);     iAs = iA >> (Nb)
    iBb = iB % (2**Nb);     iBs = iB >> (Nb)
    iH = iAb ^ iBb + (iBs << (Nb)) + (iAs << (Nb + Ns))
    dist = [];
    for j in range(len(relay_const)):
      iAp = j >> (Nb+Ns) 
      iBp = j % (2**(Nb+Ns))
      iAbp = iAp % (2**Nb);     iAsp = iAp >> (Nb)
      iBbp = iBp % (2**Nb);     iBsp = iBp >> (Nb)
      iHp = iAbp ^ iBbp + (iBsp << (Nb)) + (iAsp << (Nb + Ns))
#      print i,':',iH, iA, iB, iAb, iAs, iBb, iBs
#      print j,':',iHp, iAp, iBp, iAbp, iAsp, iBbp, iBsp, '\n'
      if iH != iHp:
        dist.append(dist_full[i,j])
    dists.append(dist)
  distances = np.asarray(dists)
#  print np.vstack(distances)
  return 1., np.min(distances)

### Throughputs evaluation

# The BC stage is supposed to reliable carry N bits
def Throughput_Eval_range(Nb, Ns, N, SNR_MAC_vals, SNR_HSI_vals, mapping='XOR', len_frame=2e2):
  if N < Nb + 2 * Ns:
    return 0
  else:
    if mapping == 'XOR':
      (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h = 1)
    elif mapping == 'MS':
      (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns, h = 1)
    if Nb > 0: # H-SI link
      (k_HSI, rho2_HSI) = eval_num_neigh(basic_part) 
      sigma2w_HSI = np.asarray([SNR2sigma2w(i) for i in SNR_HSI_vals])
      P_HSI = 1-pair_wise_err(k_HSI, sigma2w_HSI, rho2_HSI) 
    else:
      P_HSI = np.ones(len(SNR_HSI_vals))
    (k_MAC, rho2_MAC) = eval_num_neigh(relay_const, alpha) # MAC min - dist
    sigma2w_MAC = np.asarray([SNR2sigma2w(i) for i in SNR_MAC_vals])
    P_MAC = 1-pair_wise_err(k_MAC, sigma2w_MAC, rho2_MAC)
    res = np.asarray([[(i*j ) ** len_frame * (Nb + Ns) for i in P_MAC] for j in P_HSI])
    return res

  
def Throughput_Eval(Nb, Ns, N, SNR_MAC, SNR_HSI, mapping='XOR', len_frame=2e2): # N ... number of bits in BC stage
  if N < Nb + 2 * Ns:
    return 0
  else:
    if mapping == 'XOR':
      (sourceA_const, sourceB_const, basic_part, relay_const) = const_design_XOR(Nb, Ns, h = 1)
    elif mapping == 'MS':
      (sourceA_const, sourceB_const, basic_part, relay_const) = const_design_MS(Nb, Ns, h = 1)
    if Nb > 0: # H-SI link
      (k_HSI, rho2_HSI) = eval_num_neigh(basic_part) 
      sigma2w_HSI = SNR2sigma2w(SNR_HSI)
      P_HSI = 1-pair_wise_err(k_HSI, sigma2w_HSI, rho2_HSI)
    else:
      P_HSI = 1
    (k_MAC, rho2_MAC) = eval_num_neigh(relay_const) # MAC min - dist
    sigma2w_MAC = SNR2sigma2w(SNR_MAC)
    P_MAC = 1-pair_wise_err(k_MAC, sigma2w_MAC, rho2_MAC)
    return (P_MAC * P_HSI) ** len_frame * (Nb + Ns)

# Evaluate throughput for all constellations, where N-QAM const is assumed for BC stage
def Throughput_Eval_range_all(Nb, Ns, SNR_MAC_vals, SNR_HSI_vals, SNR_BC_vals, mapping='XOR',len_frame=2e2, h=1., D='both'):
  N = Nb + 2 * Ns
  constR, alpha_BC = QAM(N)
  if mapping == 'XOR':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)
  elif mapping == 'MS':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns, h)
  if Nb > 0: # H-SI link
    (k_HSI, rho2_HSI) = eval_num_neigh(basic_part) 
    sigma2w_HSI = np.asarray([SNR2sigma2w(i) for i in SNR_HSI_vals])
    P_HSI = 1-pair_wise_err(k_HSI, sigma2w_HSI, rho2_HSI) 
  else:
    P_HSI = np.ones(len(SNR_HSI_vals))
  # MAC Channel
#  if D == 'both': # Error rate in both destinations
  if h==1:
    (k_MAC, rho2_MAC) = eval_num_neigh(relay_const, alpha) # MAC min - dist
  else:
#    print Nb, Ns, mapping, h, relay_const, alpha
    (k_MAC, rho2_MAC) = eval_num_neigh_gen(Nb, Ns, mapping, h) # MAC min - dist
#  if D == 'single': # Error rate in single destination
#    (k_MAC, rho2_MAC) = eval_num_neigh(sourceA_const, alpha) # MAC min - dist
  sigma2w_MAC = np.asarray([SNR2sigma2w(i) for i in SNR_MAC_vals])
  P_MAC = 1-pair_wise_err(k_MAC, sigma2w_MAC, rho2_MAC)

  # BC Channel
  (k_BC, rho2_BC) = eval_num_neigh(constR, alpha_BC) # BC min - dist
  sigma2w_BC = np.asarray([SNR2sigma2w(i) for i in SNR_BC_vals])
  P_BC = 1-pair_wise_err(k_BC, sigma2w_BC, rho2_BC)
  
  if D == 'both': # Error rate in both destinations
    res = np.asarray([[[(i*j*j*k*k) ** len_frame * (Nb + Ns) for i in P_MAC] for j in P_HSI] for k in P_BC])
#    res = np.asarray([[[(i*j*k) ** len_frame * (Nb + Ns) for i in P_MAC] for j in P_HSI] for k in P_BC])
  if D == 'single': # Error rate in single destination
    res = np.asarray([[[(i*j*k) ** len_frame * (Nb + Ns) for i in P_MAC] for j in P_HSI] for k in P_BC])
  return res/2. # Changed against ICC 

def Throughput_Eval_all(Nb, Ns, SNR_MAC, SNR_HSI, SNR_BC, mapping='XOR', len_frame=2e2, h=1.): # N ... number of bits in BC stage
  N = Nb + 2 * Ns
  constR, alpha_BC = QAM(N)
  if mapping == 'XOR':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)
  elif mapping == 'MS':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns, h)
  if Nb > 0: # H-SI link
    (k_HSI, rho2_HSI) = eval_num_neigh(basic_part) 
    sigma2w_HSI = SNR2sigma2w(SNR_HSI)
    P_HSI = 1-pair_wise_err(k_HSI, sigma2w_HSI, rho2_HSI)
  else:
    P_HSI = 1
#  print len(relay_const)
  if h == 1:
    (k_MAC, rho2_MAC) = eval_num_neigh(relay_const, alpha) # MAC min - dist
  else:
    (k_MAC, rho2_MAC) = eval_num_neigh_gen(Nb, Ns, mapping, h)
  sigma2w_MAC = SNR2sigma2w(SNR_MAC)
  P_MAC = 1-pair_wise_err(k_MAC, sigma2w_MAC, rho2_MAC)
  (k_BC, rho2_BC) = eval_num_neigh(constR, alpha_BC) # BC min - dist
  sigma2w_BC = SNR2sigma2w(SNR_BC)
  P_BC = 1-pair_wise_err(k_BC, sigma2w_BC, rho2_BC)
  return 1./2 * (P_MAC * P_HSI * P_BC) ** len_frame * (Nb + Ns)

# Reference evaluation for JDF (Nb=0, Ns) and providing NC in relay
def Throughput_Eval_all_ref(Ns, SNR_MAC, SNR_HSI, SNR_BC, mapping='XOR', len_frame=2e2, h=1.): # N ... number of bits in BC stage
  N = Ns
  constR, alpha_BC = QAM(N) # Minimal map
  if mapping == 'XOR':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(0, Ns, h)
  elif mapping == 'MS':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(0, Ns, h)
  (k_HSI, rho2_HSI) = eval_num_neigh(sourceA_const) # Whole constellation as basic part
  sigma2w_HSI = SNR2sigma2w(SNR_HSI)
  P_HSI = 1-pair_wise_err(k_HSI, sigma2w_HSI, rho2_HSI)
#  print len(relay_const)
  if h == 1:
    (k_MAC, rho2_MAC) = eval_num_neigh(relay_const, alpha) # MAC min - dist
  else:
    (k_MAC, rho2_MAC) = eval_num_neigh_gen(Nb,Ns, mapping, h)
  sigma2w_MAC = SNR2sigma2w(SNR_MAC)
  P_MAC = 1-pair_wise_err(k_MAC, sigma2w_MAC, rho2_MAC)
  (k_BC, rho2_BC) = eval_num_neigh(constR, alpha_BC) # BC min - dist
  sigma2w_BC = SNR2sigma2w(SNR_BC)
  P_BC = 1-pair_wise_err(k_BC, sigma2w_BC, rho2_BC)
  return (P_MAC * P_HSI * P_BC) ** len_frame * (Nb + Ns)

def Throughput_Eval_range_all_ref(Ns, SNR_MAC_vals, SNR_HSI_vals, SNR_BC_vals, mapping='XOR',len_frame=2e2, h=1., D='both'):
  N =  Ns
  Nb = 0
  constR, alpha_BC = QAM(N)
  if mapping == 'XOR':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)
  elif mapping == 'MS':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns, h)
  (k_HSI, rho2_HSI) = eval_num_neigh(sourceA_const) 
  sigma2w_HSI = np.asarray([SNR2sigma2w(i) for i in SNR_HSI_vals])
  P_HSI = 1-pair_wise_err(k_HSI, sigma2w_HSI, rho2_HSI) 
  # MAC Channel
#  if D == 'both': # Error rate in both destinations
  (k_MAC, rho2_MAC) = eval_num_neigh(relay_const, alpha) # MAC min - dist
#  if D == 'single': # Error rate in single destination
#    (k_MAC, rho2_MAC) = eval_num_neigh(sourceA_const, alpha) # MAC min - dist
  sigma2w_MAC = np.asarray([SNR2sigma2w(i) for i in SNR_MAC_vals])
  P_MAC = 1-pair_wise_err(k_MAC, sigma2w_MAC, rho2_MAC)

  # BC Channel
  (k_BC, rho2_BC) = eval_num_neigh(constR, alpha_BC) # BC min - dist
  sigma2w_BC = np.asarray([SNR2sigma2w(i) for i in SNR_BC_vals])
  P_BC = 1-pair_wise_err(k_BC, sigma2w_BC, rho2_BC)
  
  if D == 'both': # Error rate in both destinations
    res = np.asarray([[[(i*j*j*k*k) ** len_frame * (Nb + Ns) for i in P_MAC] for j in P_HSI] for k in P_BC])
#    res = np.asarray([[[(i*j*k) ** len_frame * (Nb + Ns) for i in P_MAC] for j in P_HSI] for k in P_BC])
  if D == 'single': # Error rate in single destination
    res = np.asarray([[[(i*j*k) ** len_frame * (Nb + Ns) for i in P_MAC] for j in P_HSI] for k in P_BC])
  return res

def Throughput_Eval_range_all_ref_3_slot_min_map(N, SNR_MAC_vals, SNR_HSI_vals, SNR_BC_vals, mapping='XOR',len_frame=2e2, h=1., D='both'):
  const, alpha = QAM(N) * np.sqrt(3./2)
  (kc, rho2) = eval_num_neigh(const, alpha) # BC min - dist
  sigma2w_HSI = np.asarray([SNR2sigma2w(i) for i in SNR_HSI_vals])
  P_HSI = 1-pair_wise_err(kc, sigma2w_HSI, rho2) 
  # MAC Channel
  sigma2w_MAC = np.asarray([SNR2sigma2w(i) for i in SNR_MAC_vals])
  P_MAC = 1-pair_wise_err(kc, sigma2w_MAC, rho2)

  # BC Channel
  sigma2w_BC = np.asarray([SNR2sigma2w(i) for i in SNR_BC_vals])
  P_BC = 1-pair_wise_err(kc, sigma2w_BC, rho2)
  
  if D == 'both': # Error rate in both destinations
    res = 1./3 * np.asarray([[[(i*i*j*j*k*k) ** len_frame * N for i in P_MAC] for j in P_HSI] for k  in P_BC]) 
#    res = np.asarray([[[(i*j*k) ** len_frame * (Nb + Ns) for i in P_MAC] for j in P_HSI] for k in P_BC])
  if D == 'single': # Error rate in single destination
    res = 1./3 * np.asarray([[[(i*i*j*k) ** len_frame * N for i in P_MAC] for j in P_HSI] for k in P_BC])
  return res

def Throughput_Eval_range_all_ref_3_slot_JDF(N, SNR_MAC_vals, SNR_HSI_vals, SNR_BC_vals, mapping='XOR',len_frame=2e2, h=1., D='both'):
  # MAC Channel
  const, alpha = QAM(N) * np.sqrt(3./2)
  (kMAC, rho2) = eval_num_neigh(const, alpha) # BC min - dist
  sigma2w_MAC = np.asarray([SNR2sigma2w(i) for i in SNR_MAC_vals])
  P_MAC = 1-pair_wise_err(kMAC, sigma2w_MAC, rho2)

  # BC Channel
  constBC, alphaBC = QAM(N**2) * np.sqrt(3./2)
  (kBC, rho2BC) = eval_num_neigh(constBC, alphaBC) # BC min - dist
  sigma2w_BC = np.asarray([SNR2sigma2w(i) for i in SNR_BC_vals])
  P_BC = 1-pair_wise_err(kBC, sigma2w_BC, rho2)
  
   
  if D == 'both': # Error rate in both destinations
    res = 1./3 * np.asarray([[[(i*i*k*k) ** len_frame * N for i in P_MAC] for j in SNR_HSI_vals] for k  in P_BC]) 
#    res = np.asarray([[[(i*j*k) ** len_frame * (Nb + Ns) for i in P_MAC] for j in P_HSI] for k in P_BC])
  if D == 'single': # Error rate in single destination
    res = 1./3 * np.asarray([[[(i*i*k) ** len_frame * N for i in P_MAC] for j in SNR_HSI_vals] for k in P_BC])
  return res

# Some older functions
# Evaluate Throughput (2 bits in source)
def throughput_JDF(SNR_MAC, SNR_BC, len_frame = 2e2):
  P_MAC = (1 - eval_analytic_BER_HDF8ex(SNR_MAC)) ** len_frame
  P_BC = (1 - eval_analytic_BER_16QAM(SNR_BC)) ** len_frame
  return P_MAC * P_BC * 2

def throughput_HDF_ex(SNR_MAC, SNR_BC, SNR_HSI, len_frame = 2e2):
  P_MAC = (1 - eval_analytic_BER_HDF8ex(SNR_MAC)) ** len_frame
#  P_BC = (1 - eval_analytic_BER_8PSK(SNR_BC)) ** len_frame
  P_BC = (1 - eval_analytic_BER_8QAM(SNR_BC)) ** len_frame
  P_HSI = (1 - eval_analytic_BER_site_link(SNR_HSI)) ** len_frame
  return P_MAC * P_BC * P_HSI * 2

def throughput_HDF_min(SNR_MAC, SNR_BC, SNR_HSI, len_frame = 2e2):
  P_MAC = (1 - eval_analytic_BER_QPSK_hierarchical(SNR_MAC)) ** len_frame
  P_BC = (1 - eval_analytic_BER_QPSK(SNR_BC)) ** len_frame
  P_HSI = (1 - eval_analytic_BER_QPSK(SNR_HSI)) ** len_frame
  return P_MAC * P_BC * P_HSI * 2

## Evaluate Throughput (3 bits in source)
def throughput3_JDF(SNR_MAC, SNR_BC, len_frame = 2e2):
  P_MAC = 1 #(1 - eval_analytic_BER_HDF8ex(SNR_MAC)) ** len_frame
  P_BC = (1 - eval_analytic_BER_64QAM(SNR_BC)) ** len_frame
  return P_MAC * P_BC * 3

# 1 bit by site link
def throughput3_HDF_ex1(SNR_MAC, SNR_BC, SNR_HSI, len_frame = 2e2):
  P_MAC = 1#(1 - eval_analytic_BER_HDF8ex(SNR_MAC)) ** len_frame
  P_BC = (1 - eval_analytic_BER_32QAM(SNR_BC)) ** len_frame
  P_HSI = (1 - eval_analytic_BER_site_link_1bit(SNR_HSI)) ** len_frame
  return P_MAC * P_BC * P_HSI * 3

# 2 bits by site link
def throughput3_HDF_ex2(SNR_MAC, SNR_BC, SNR_HSI, len_frame = 2e2):
  P_MAC = 1#(1 - eval_analytic_BER_HDF8ex(SNR_MAC)) ** len_frame
  P_BC = (1 - eval_analytic_BER_16QAM(SNR_BC)) ** len_frame
  P_HSI = (1 - eval_analytic_BER_site_link_2bits(SNR_HSI)) ** len_frame
  return P_MAC * P_BC * P_HSI * 3

def throughput3_HDF_min(SNR_MAC, SNR_BC, SNR_HSI, len_frame = 2e2):
  P_MAC = 1#(1 - eval_analytic_BER_QPSK_hierarchical(SNR_MAC)) ** len_frame
  P_BC = (1 - eval_analytic_BER_8QAM(SNR_BC)) ** len_frame
  P_HSI = (1 - eval_analytic_BER_8QAM(SNR_HSI)) ** len_frame
  return P_MAC * P_BC * P_HSI * 3

# Find solution for monotonically descreasing function
def find_solution(target, function, error, init_val = 10, init_step = 5., max_iter=50):
  x = init_val
  act = function(x)
  step = init_step
  direct = 0
  count = 0
  while ((np.abs(act - target) >=error) and (count < max_iter)):
    if (act - target > 0):
      if (direct == -1):
	step = step / 2
      x += step
      direct = 1
    if (act - target < 0):
      if (direct == 1):
	step = step / 2
      x -= step
      direct = -1
    count += 1
    act = function(x)
#    print x, act, step
  return x

## Design of constallations
# Recursive function to achieve constallations given by certain levels
def find_levels(i, levels, const, const_el):
  if (len(levels) == 0):
    return np.array([0]) # Empty constellation --- zero point
  else:
    const_out = np.asarray([c + levels[i] * el for c in const for el in const_el])
#    print np.sort(const_out)
    i += 1
    if (i < len(levels)):
      return find_levels(i, levels, const_out, const_el)
    else:
      return const_out

# Constellation with bitwise XOR hierarchical mapping
def eval_levels_XOR(Nb, Ns):  # Evaluate BPSK levels for given Nb, Ns -- complex channel
  odd = np.mod(Nb, 2)  
  Ls = np.asarray([2**i for i in np.arange(0, Ns)]) # superposed levels
  Lbt = 2**Ns * np.asarray([np.asarray([3**i, 1j*3**i]) for i in np.arange(0, Nb/2)]).flatten() # basic levels
#  print np.abs(Ls), np.abs(Lb)
  if odd == 1:
    Lb = np.hstack([Lbt, 2**Ns * 3**((Nb-1)/2)])
  else: 
    Lb = Lbt
  return (Lb, Ls) # Basic and superposed levels

def const_design_XOR(Nb, Ns, h = 1.): # Relative Channel Parameterization
  (Lb, Ls) = eval_levels_XOR(Nb, Ns)
  en = np.sum(np.abs(Lb)**2) + np.sum(np.abs(Ls)**2)
  alpha = 1./np.sqrt(en)
  sourceA_const = find_levels(0, np.hstack([Ls, Lb]), [0], [-1., 1.])*alpha
  sourceB_const = find_levels(0, np.hstack([1j*Ls, Lb]), [0], [-1., 1.])*alpha
  basic_part = find_levels(0,Lb, [0], [-1., 1.])*alpha
  relay_const = np.asarray([x + y * h for x in sourceA_const for y in sourceB_const])
  return (sourceA_const, sourceB_const, basic_part, relay_const, alpha) 

# Constellation with bitwise mod-sum hierarchical mapping
def eval_levels_MS(Nb, Ns):  # Evaluate BPSK levels for given Nb, Ns -- complex channel
  odd = np.mod(Nb, 2)  
  Ls = np.asarray([2**i for i in np.arange(0, Ns)]) # superposed levels
  if odd == 1:
    Lbt = np.hstack([np.array([2**((Nb-1)/2)]), \
     np.asarray([2**i for i in np.arange(Nb/2-1,-1,-1)]).flatten()])
  else:
    Lbt = np.asarray([2**i for i in np.arange(Nb/2-1,-1,-1)]).flatten()
  Lb = 2**Ns * np.hstack([Lbt,\
  np.asarray([1j*2**i for i in np.arange(Nb/2-1,-1,-1)]).flatten()]) # basic levels
#   
#   Lb = 2**Ns * np.hstack([np.asarray([2**i for i in np.arange(0, Nb/2+1).flatten()]),\
#                          np.asarray([1j * 2**i for i in np.arange(0, Nb/2).flatten()])])
# else: 
#   Lb = 2**Ns * np.hstack([np.asarray([2**i for i in np.arange(0, Nb/2).flatten()]),\
#                          np.asarray([1j * 2**i for i in np.arange(0, Nb/2).flatten()])])
  return (Lb, Ls) # Basic and superposed levels

def const_design_MS(Nb, Ns, h = 1.):
  (Lb, Ls) = eval_levels_MS(Nb, Ns)
  en = np.sum(np.abs(Lb)**2) + np.sum(np.abs(Ls)**2)
  alpha = 1./np.sqrt(en)
  sourceA_const = find_levels(0, np.hstack([Ls, Lb]), [0], [-1., 1.])*alpha
  sourceB_const = find_levels(0, np.hstack([1j*Ls, Lb]), [0], [-1., 1.])*alpha
  basic_part = find_levels(0,Lb, [0], [-1., 1.])*alpha
  relay_const = np.asarray([x + y * h  for x in sourceA_const for y in sourceB_const])
  return (sourceA_const, sourceB_const, basic_part, relay_const, alpha) 

def min_dist_eval(Nb, Ns, phi = 0, mapping = 'XOR'):
  # Evaluate minimal distance with repect to rotation
  if mapping == 'XOR':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, phi)
    ind_const = np.asarray([s1* 2**(Nb + Ns) + s2 * 2**(Nb) + (b1^b2) \
              for s1 in range(0, 2**Ns) for b1 in range(0, 2**Nb) \
              for s2 in range(0, 2**Ns) for b2 in range(0, 2**Nb)] )
  elif mapping == 'MS':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns, phi)
    ind_const = np.asarray([s1* 2**(Nb + Ns) + s2 * 2**(Nb) + (np.mod(b1+b2, 2**Nb)) \
              for s1 in range(0, 2**Ns) for b1 in range(0, 2**Nb) \
              for s2 in range(0, 2**Ns) for b2 in range(0, 2**Nb)] )
  dist = np.asarray([np.abs(relay_const[i] - relay_const[j])**2 \
         for i in range(0, len(relay_const)) for j in range(i+1, len(relay_const))\
         if relay_const[ind_const[i]] != relay_const[ind_const[j]]])
  return np.min(dist)
 

def complex_info(Nb, Ns, mapping='XOR'):
  if mapping == 'XOR':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns)
  elif mapping == 'MS':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns)
  plt.plot(np.real(sourceA_const), np.imag(sourceA_const),'x',ms=12)
  plt.plot(np.real(sourceB_const), np.imag(sourceB_const),'go',ms=10)
  plt.plot(np.real(basic_part), np.imag(basic_part),'cd',ms=10)
  plt.plot(np.real(relay_const), np.imag(relay_const),'ro',ms=8)
  plt.legend(('Sa', 'Sb', 'Basic', 'Superposed'),numpoints=1)
  dr = np.str(eval_num_neigh(relay_const)[1])
  dA = np.str(eval_num_neigh(sourceA_const)[1])
  dB = np.str(eval_num_neigh(sourceB_const)[1])
  db = np.str(eval_num_neigh(basic_part)[1])
  plt.show()
  print dr, dA,dB, db


def QAM(N): # QAM constellation
  odd = N % 2
  base = (N - odd) / 2
  min_Re = -2**(base+odd) + 1
  max_Re = 2**(base+odd) - 1
  min_Im = -2**base + 1
  max_Im = 2**base - 1
  const = np.asarray([i + 1j*j  for i in np.arange(min_Re, max_Re+1, 2)\
                                   for j in np.arange(min_Im, max_Im+1, 2)])
  scale = np.sqrt((np.abs(const)**2).sum()/float(2**N))
  return const / scale, 1./scale

def const_superposed(Nb, Ns, mapping):
  if mapping == 'XOR':
    (Lb, Ls) = eval_levels_XOR(Nb, Ns)
  elif mapping == 'MS':
    (Lb, Ls) = eval_levels_MS(Nb, Ns)
  en = np.sum(np.abs(Lb)**2) + np.sum(np.abs(Ls)**2)
  alpha = 1./np.sqrt(en)
  return find_levels(0, Ls, [0], [-1., 1.])*alpha


## Capacities evaluation
# P2P capacity
def cap_eval(SNR, const, M):
  sigma2w = SNR2sigma2w(SNR)
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
  print Hx, Hxd
  return Hx - Hxd;

## Simulation Evaluation
def Error_Check_Relay(dH, dA, dB, Nb, Ns, mapping):
  dAb = dA % (2**Nb) # Separation to basic and superposed part
  dBb = dB % (2**Nb)
  dAs = dA / (2**Nb)
  dBs = dB / (2**Nb)
  if mapping == 'XOR':
    dR = dAs * 2**(Nb + Ns) + dBs * 2**(Nb) + (dAb^dBb)
  elif mapping == 'MS':
    dR = dAs * 2**(Nb + Ns) + dBs * 2**(Nb) + ((dAb+dBb) % (2**Nb))
  nerr = len(np.nonzero(dR != dH)[0])
  return nerr/float(len(dA))

def Eval_Data(Nb, Ns, node, mapping, dH, d):
  dbh = dH % (2**(Nb)) # NCed basic parts
  if mapping == 'XOR':
    db = d ^ dbh # basic part
  elif mapping == 'MS':
    db = ((dbh - d) % (2**Nb)) # basic part
  dBs = (dH >> Nb)  % (2**Ns)
  dAs = (dH >> (Nb + Ns))
  if node == 'DA':
    return db + (dAs << Nb)
  elif node == 'DB':
    return db + (dBs << Nb)


def Numerical_Simulation(SNR_MAC, SNR_BC, SNR_HSI, frame_len, Nb, Ns, mapping, num_frames, D):
  # Channel parameters 
  hA = 1.;hB = hA;hRA = 1.;hRB = 1.;hAB = 1.;hBA = 1.
  sigma2w_MAC = SNR2sigma2w(SNR_MAC)
  sigma2w_BC = SNR2sigma2w(SNR_BC)
  sigma2w_HSI = SNR2sigma2w(SNR_HSI)
  if mapping == 'XOR':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h = 1)
    ind_const = np.asarray([s1* 2**(Nb + Ns) + s2 * 2**(Nb) + (b1^b2) \
              for s1 in range(0, 2**Ns) for b1 in range(0, 2**Nb) \
              for s2 in range(0, 2**Ns) for b2 in range(0, 2**Nb)] )
  elif mapping == 'MS':
    (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns, h = 1)
    ind_const = np.asarray([s1* 2**(Nb + Ns) + s2 * 2**(Nb) + ((b1+b2) % (2**Nb)) \
              for s1 in range(0, 2**Ns) for b1 in range(0, 2**Nb) \
              for s2 in range(0, 2**Ns) for b2 in range(0, 2**Nb)] )
  const_SP = const_superposed(Nb, Ns, mapping)
  nerr = np.zeros(num_frames)
  nerr_R = np.zeros(num_frames)
  for i in range(num_frames):
  # Source A
    dA = np.random.randint(2**Nb * 2**Ns, size = frame_len)
    sA = sourceA_const[dA]
  # Source B
    dB = np.random.randint(2**Nb * 2**Ns, size = frame_len)
    sB = sourceB_const[dB]
  # MAC channel
    w_MAC = np.sqrt(sigma2w_MAC/2)*(np.random.normal(size=frame_len) + 1j*np.random.normal(size=frame_len))
    x = hA * sA + hB * sB + w_MAC
  # Relay Processing
    dist = np.asarray([np.abs(x - hA * A - hB * B)**2 \
              for A in sourceA_const for B in sourceB_const] )
    dH = ind_const[np.argmin(dist, axis = 0)]
    N = 2*Ns+Nb
    BC_const,alpha = QAM(N)
    sR = BC_const[dH]
  #  print 'Relay Pbe:', Error_Check_Relay(dH, dA, dB, Nb, Ns, mapping)
    nerr_R[i] = Error_Check_Relay(dH, dA, dB, Nb, Ns, mapping)

  # BC channel
    w_BC1 = np.sqrt(sigma2w_BC/2)*(np.random.normal(size=frame_len) + 1j*np.random.normal(size=frame_len))
    w_BC2 = np.sqrt(sigma2w_BC/2)*(np.random.normal(size=frame_len) + 1j*np.random.normal(size=frame_len))
    yA = hRA * sR + w_BC1 # dest A
    yB = hRB * sR + w_BC2 # dest B 
  # H-SI channels
    w_HSI1 = np.sqrt(sigma2w_HSI/2)*(np.random.normal(size=frame_len) + 1j*np.random.normal(size=frame_len))
    w_HSI2 = np.sqrt(sigma2w_HSI/2)*(np.random.normal(size=frame_len) + 1j*np.random.normal(size=frame_len))
    zA = hAB * sA + w_HSI1 # dest B
    zB = hBA * sB + w_HSI2 # dest A
  
  # Destination A
    dist = np.asarray([np.abs(yA - hRA*s0)**2 for s0 in BC_const])
    dHA_est = np.argmin(dist, axis = 0)
    db_estA = dHA_est % (2**(Nb))
    dBs_estA = (dHA_est >> Nb)  % (2**Ns)
    dAs_estA = (dHA_est >> (Nb + Ns))
  #  print 'Pbe of dBs, dAs: ', len(np.nonzero((dB >> Nb) != dBs_estA)[0])/float(frame_len), \
  #                             len(np.nonzero((dA >> Nb) != dAs_estA)[0])/float(frame_len), \
  #                             len(np.nonzero(((dA^dB) % 2**Nb) != db_estA)[0])/float(frame_len)
    sB_recA = const_SP[dBs_estA] * 1j # recovered superposed part of sB
  #  plt.plot(np.real(sB_recA), np.imag(sB_recA), 'x')
  #  plt.plot(np.real(zB), np.imag(zB), 'o')
  #  plt.plot(np.real(sB), np.imag(sB), 'd')
    dist = np.asarray([np.abs(zB - hBA * sB_recA - hBA*s0)**2 for s0 in basic_part] )
    dB_b_estA = np.argmin(dist, axis = 0)
  #  print 'Basic dB in Dest A Pbe: ', len(np.nonzero((dB % (2**Nb)) != dB_b_estA)[0])/float(frame_len)
    est_dA = Eval_Data(Nb, Ns, 'DA', mapping, dHA_est, dB_b_estA)
    nerr[i] += len(np.nonzero(dA != est_dA)[0]) # Symbol Error in Dest A
  #  print 'Dest A Pbe: ', len(np.nonzero(dA != est_dA)[0])/float(frame_len)
  #  plt.show()
  
  
  # Destination B
    dist = np.asarray([np.abs(yB - hRA*s0)**2 for s0 in BC_const])
    dHB_est = np.argmin(dist, axis = 0)
    db_estB = dHB_est % (2**(Nb))
    dBs_estB = (dHB_est >> Nb)  % (2**Ns)
    dAs_estB = (dHB_est >> (Nb + Ns))
    sA_recB = const_SP[dAs_estB]  # recovered superposed part of sA
    dist = np.asarray([np.abs(zA - hBA * sA_recB - hBA*s0)**2 for s0 in basic_part] )
    dA_b_estB = np.argmin(dist, axis = 0)
    est_dB = Eval_Data(Nb, Ns, 'DB', mapping, dHB_est, dA_b_estB)
    if D == 'both':
      nerr[i] += len(np.nonzero(dB != est_dB)[0]) # Symbol Error in Dest B
  #  print 'Dest B Pbe: ', len(np.nonzero(dB != est_dB)[0])/float(frame_len)

  return nerr

def Throughput_Eval_Numerical(Nb, Ns, mapping, len_Frame, num_frames, SNR_HSI, SNR_MAC, SNR_BC, D='both'):
  out = np.zeros([len(SNR_HSI), len(SNR_MAC), len(SNR_BC)], float)
  for i in range(len(SNR_MAC)):  
    for j in range(len(SNR_HSI)):  
      for k in range(len(SNR_BC)):  
        nerr = Numerical_Simulation(SNR_MAC[i], SNR_BC[k], SNR_HSI[j], len_Frame, Nb, Ns, mapping, num_frames, D)
        out[j, i, k] = (Nb + Ns) * (1-(sum(nerr > 0) / float(num_frames))) # 1 - FER
  return out


## Main run
if __name__ == '__main__':
#HDF_consts_design(4,1)

  SNR_vals = np.arange(0,30,3)
#SNR_vals = [20]
  N = 1e2


  sQPSK = 0
  if sQPSK:
    BER = np.asarray([eval_analytic_BER_QPSK(SNR) for SNR in SNR_vals])
    BERs = np.asarray(eval_numerical_BER_QPSK(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER,'ko-')
    plt.semilogy(SNR_vals,BERs,'bx-')

  sQPSK_HDF = 0
  if sQPSK_HDF:
    BER_HDF = np.asarray([eval_analytic_BER_QPSK_hierarchical(SNR) for SNR in SNR_vals])
    BER_HDFs = np.asarray(eval_numerical_BER_QPSK_HDF(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER_HDF,'ko-')
    plt.semilogy(SNR_vals,BER_HDFs,'bx-')

  sHSI_link_3 = 0
  if sHSI_link_3:
    BER_HDF1 = np.asarray([eval_analytic_BER_site_link_1bit(SNR) for SNR in SNR_vals])
    BER_HDF2 = np.asarray([eval_analytic_BER_site_link_2bits(SNR) for SNR in SNR_vals])
    BER_HDF1s = np.asarray(eval_numerical_BER_site_link_1bit(SNR_vals, N), dtype=float) / N
    BER_HDF2s = np.asarray(eval_numerical_BER_site_link_2bits(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER_HDF1,'ko-')
    plt.semilogy(SNR_vals,BER_HDF2,'bx-')
    plt.semilogy(SNR_vals,BER_HDF1s,'ro-')
    plt.semilogy(SNR_vals,BER_HDF2s,'cx-')

  sQAM16 = 0
  if sQAM16:
    BER_16QAM = np.asarray([eval_analytic_BER_16QAM(SNR) for SNR in SNR_vals])
    BER_16QAMs = np.asarray(eval_numerical_BER_16QAM(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER_16QAM,'ko-')
    plt.semilogy(SNR_vals,BER_16QAMs,'bx-')

  sQAM32 = 0
  if sQAM32:
    QAM32 = np.asarray([i + 1j * j for i in np.arange(-7,9,2) for j in np.arange(-3,5,2)])/np.sqrt(26.)
    BER_32QAM = np.asarray([eval_analytic_BER_32QAM(SNR) for SNR in SNR_vals])
    BER_32QAMs = np.asarray(eval_numerical_BER_general_mod(SNR_vals, QAM32, N), dtype=float) / N
#  BER_32QAMa = np.asarray([pair_wise_err(SNR, QAM32) for SNR in SNR_vals])
    plt.semilogy(SNR_vals,BER_32QAM,'ko-')
    plt.semilogy(SNR_vals,BER_32QAMs,'bx-')
    plt.semilogy(SNR_vals,BER_32QAMa,'rd-')

  sQAM8 = 0
  if sQAM8:
    BER_8QAM = np.asarray([eval_analytic_BER_8QAM(SNR) for SNR in SNR_vals])
    BER_8QAMs = np.asarray(eval_numerical_BER_8QAM(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER_8QAM,'ko-')
    plt.semilogy(SNR_vals,BER_8QAMs,'bx-')
    plt.savefig('./fig.png')

  sQAM64 = 0
  if sQAM64:
    BER_64QAM = np.asarray([eval_analytic_BER_64QAM(SNR) for SNR in SNR_vals])
    BER_64QAMs = np.asarray(eval_numerical_BER_64QAM(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER_64QAM,'ko-')
    plt.semilogy(SNR_vals,BER_64QAMs,'bx-')

  sQAM256 = 0
  if sQAM256:
    BER_256QAM = np.asarray([eval_analytic_BER_256QAM(SNR) for SNR in SNR_vals])
    BER_256QAMs = np.asarray(eval_numerical_BER_256QAM(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER_256QAM,'ko-')
    plt.semilogy(SNR_vals,BER_256QAMs,'bx-')

  sPSK8 = 0
  if sPSK8:
    BER_8PSK = np.asarray([eval_analytic_BER_8PSK(SNR) for SNR in SNR_vals])
    BER_8PSKs = np.asarray(eval_numerical_BER_8PSK(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER_8PSK,'ko-')
    plt.semilogy(SNR_vals,BER_8PSKs,'bx-')

  sHDF_ext = 0
  if  sHDF_ext:
    BER_8HDF = np.asarray([eval_analytic_BER_HDF8ex(SNR) for SNR in SNR_vals])
    BER_8HDFs = np.asarray(eval_numerical_BER_exHDF(SNR_vals, N), dtype=float) / N
    BER_16fulls = np.asarray(eval_numerical_BER_fullHDF(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER_8HDF,'ko-')
    plt.semilogy(SNR_vals,BER_8HDFs,'bx-')
    plt.semilogy(SNR_vals,BER_16fulls,'cd-')

  sHDF_HSI = 0
  if sHDF_HSI:
    BER_8HDF_SI = np.asarray([eval_analytic_BER_site_link(SNR) for SNR in SNR_vals])
    BER_8HDF_SIs = np.asarray(eval_numerical_BER_site_link(SNR_vals, N), dtype=float) / N
    plt.semilogy(SNR_vals,BER_8HDF_SI,'ko-')
    plt.semilogy(SNR_vals,BER_8HDF_SIs,'bx-')

# Overall BER (analytical)
  soverall = 0
  if soverall:
    BER = np.asarray([eval_analytic_BER_QPSK(SNR) for SNR in SNR_vals])
    BER_HDF = np.asarray([eval_analytic_BER_QPSK_hierarchical(SNR) for SNR in SNR_vals])
    BER_16QAM = np.asarray([eval_analytic_BER_16QAM(SNR) for SNR in SNR_vals])
    BER_8PSK = np.asarray([eval_analytic_BER_8PSK(SNR) for SNR in SNR_vals])
    BER_8HDF = np.asarray([eval_analytic_BER_HDF8ex(SNR) for SNR in SNR_vals])
    BER_8HDF_SI = np.asarray([eval_analytic_BER_site_link(SNR) for SNR in SNR_vals])
    plt.semilogy(SNR_vals,BER,'ko-', ms=10)
    plt.semilogy(SNR_vals,BER_16QAM,'cd-', ms=10)
    plt.semilogy(SNR_vals,BER_8PSK,'rp-', ms=10)
    plt.semilogy(SNR_vals,BER_HDF,'bx--', ms=13)
    plt.semilogy(SNR_vals,BER_8HDF,'ms--', ms=13)
    plt.semilogy(SNR_vals,BER_8HDF_SI,'g*--', ms=15)
    plt.grid()
    plt.ylim( (1e-6, 1))
    plt.legend(('QPSK','16QAM','8PSK','HDF_min', 'ex+join_MAC','ex_Site_link'))



#SNR_vals = np.asarray([find_solution(10**(-x), eval_analytic_BER_site_link, 1e-12) for x in np.arange(1,8)])
  #plt.show()
