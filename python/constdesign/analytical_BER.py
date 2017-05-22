import numpy as np
from scipy.special import erf
from scipy.special import erfc

# Function returning the value of two independent gaussian CDF(x1, x2)
def prod_gauss2D(x1, m1, x2, m2, sigma2w):
  D1 = 0.5 * (1. + erf((x1 - m1) / np.sqrt(sigma2w)))
  D2 = 0.5 * (1. + erf((x2 - m2) / np.sqrt(sigma2w)))
  return D1 * D2;

def SNR2sigma2w(SNR):
  alpha = 10**(float(SNR)/10)
  return 1.0/alpha

# Analytical BER evaluation for QPSK
def eval_analytic_BER_QPSK(SNR):
  sigma2w = SNR2sigma2w(SNR)
  return 1 - prod_gauss2D(0, -np.sqrt(1./2),0 , -np.sqrt(1./2),sigma2w)

# Analytical BER evaluation for 8PSK
def eval_analytic_BER_8PSK(SNR):
  sigma2w = SNR2sigma2w(SNR)
  Delta = 0.01
  x = np.arange(0,20,Delta)
  y = np.tan(np.pi/8) * x
  p = 0
  for i in range(0, len(x) - 1):
    LUu = prod_gauss2D(x[i], 1, y[i + 1], 0,sigma2w)
    RUu = prod_gauss2D(x[i + 1], 1, y[i + 1], 0,sigma2w)

    LUd = prod_gauss2D(x[i], 1, -y[i + 1], 0,sigma2w)
    RUd = prod_gauss2D(x[i + 1], 1, -y[i + 1], 0,sigma2w)
    p += RUu - RUd - (LUu - LUd)    
  return 1 - p

# Analytical BER evaluation for HDF with QPSK
def eval_analytic_BER_QPSK_hierarchical(SNR):
  sigma2w = SNR2sigma2w(SNR)
  QPSK = np.exp(1j * (np.pi * np.arange(0,4)/2.0 + np.pi/4))
  const_superposed = np.round(np.asarray([i + j for i in QPSK for j in QPSK]).flatten(),8)
  con_unique = np.unique(const_superposed)
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in con_unique])
  x_shape = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, np.inf])
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], u, SI_c[i,1], sigma2w) \
  for t in x_shape for u in x_shape]).flatten() for i in range(0,9)]) 
  # p(x|s), where x are received const points and s are tranmitted const points
  net2 = np.zeros(9)
  net2[0] = net[6,6] - net[6,3]# p0
  net2[1] = net[7,7]+net[7,3]-net[7,6]-net[7,4] # p1
  net2[2] = 1 +net[8,4]-net[8,7]-net[8,5] # p2
  net2[3] = net[3,3] - net[3,0] # p3
  net2[4] = net[4,4]+net[4,0]-net[4,3]-net[4,1] # p4
  net2[5] = net[5,5]+net[5,1]-net[5,4]-net[5,2] # p5
  net2[6] = net[0,0] # p6
  net2[7] = net[1,1]-net[1,0] # p7
  net2[8] = net[2,2]-net[2,1] # p8
#  print net2

  Pe = (1-net2[0])+(1-net2[2])+(1-net2[6])+(1-net2[8])
  Pe += (4-net2[4]*4) 
  Pe += 4-(2*net2[1]+2*net2[7]) 
  Pe += 4-(2*net2[3]+2*net2[5]) 
  return Pe/16

# Analytical BER evaluation for 8QAM
def eval_analytic_BER_8QAM(SNR):
  sigma2w = SNR2sigma2w(SNR)
  QAM8 = np.asarray([i + 1j * j for i in np.arange(-3,0,2) for j in np.arange(-1,0,2)]) /np.sqrt(6)
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in QAM8])
  x_shape = np.hstack([np.asarray([i for i in np.arange(-2,1,2)])/ np.sqrt(6)])
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], 0, SI_c[i,1], sigma2w) \
  for t in x_shape ]).flatten() for i in range(0,2)])
  return 1 -(net[0,0] + net[1,1] - net[1,0])/2.


# Analytical BER evaluation for 16QAM
def eval_analytic_BER_16QAM(SNR):
  sigma2w = SNR2sigma2w(SNR)
  QAM16 = np.asarray([i + 1j * j for i in np.arange(-3,5,2) for j in np.arange(-3,5,2)]) /np.sqrt(10)
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in QAM16])
  x_shape = np.hstack([np.asarray([i for i in np.arange(-2,3,2)])/ np.sqrt(10), np.inf])
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], u, SI_c[i,1], sigma2w) \
  for t in x_shape for u in x_shape]).flatten() for i in range(0,16)]) 
  # p(x|s), where x are received const points and s are tranmitted const points
#  net2 = np.zeros(16)
  net2 = np.zeros(4)
  net2[0] = net[0,0] # p0
  net2[1] = net[1,1]-net[1,0] # p1
#  net2[2] = net[2,2]-net[2,1] # p2
#  net2[3] = net[3,3]-net[3,2] # p3

  net2[2] = net[4,4] - net[4,0] # p4
  net2[3] = net[5,5]-net[5,4] - net[5, 1] + net[5, 0] # p5
#  net2[4] = net[4,4] # p4
#  net2[5] = net[5,5]-net[5,4] - net[5, 1] + net[5, 0] # p5
#  net2[6] = net[6,6]-net[6,5] - net[6, 2] + net[6, 1] # p6
#  net2[7] = net[7,7]-net[7,6] - net[7, 3] + net[7, 1] # p7

#  for i in range(0,3):
#    net2[(i * 4) +4] = net[(i * 4) +4,(i * 4) +4] - net[(i * 4) +4,(i * 4)]    # p(i * 4) +4

#    net2[(i * 4) +5] = net[(i * 4) +5,(i * 4) +5] - net[(i * 4) +5,(i * 4) +4] -\
#                       net[(i * 4) +5,(i * 4) +1] + net[(i * 4) +5,(i * 4)]    # p(i * 4) +5

#    net2[(i * 4) +6] = net[(i * 4) +6,(i * 4) +6] - net[(i * 4) +6,(i * 4) +5] -\
#                       net[(i * 4) +6,(i * 4) +2] + net[(i * 4) +6,(i * 4) +1] # p(i * 4) +6

#    net2[(i * 4) +7] = net[(i * 4) +7,(i * 4) +7] - net[(i * 4) +7,(i * 4) +6] -\
#                       net[(i * 4) +7,(i * 4) +3] + net[(i * 4) +7,(i * 4) +2] # p(i * 4) +7

  return np.sum(1.-net2)/4
#  return np.sum(1.-net2)/16

# Analytical BER evaluation for 256QAM
def eval_analytic_BER_256QAM(SNR):
  sigma2w = SNR2sigma2w(SNR)
# QAM256 = np.asarray([i + 1j * j for i in np.arange(-15,17,2) for j in np.arange(-15,17,2)])/np.sqrt(170)
  QAM256 = np.asarray([i + 1j * j  for i in np.arange(-15,0,2) for j in np.arange(-15,0,2)])/np.sqrt(170)
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in QAM256])
  x_shape = np.hstack([np.asarray([i for i in np.arange(-14,1,2)])/ np.sqrt(170)])
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], u, SI_c[i,1], sigma2w) \
  for t in x_shape for u in x_shape]).flatten() for i in range(0,64)]) 
  # p(x|s), where x are received const points and s are tranmitted const points
#  net2 = np.zeros(16)
  net2 = np.zeros(64)
  net2[0] = net[0,0] # p0
  for i in range(1,8):
    net2[i] = net[i, i] - net[i, i - 1]

  for i in range(1,8):
    act = i  * 8# act -8 -> down, act -1 -> left, act - 9 -> left down
    net2[act] = net[act, act] - net[act, act - 8] 
    for j in range(1,8):
      act = i * 8 + j
      net2[act] = net[act, act] - net[act, act - 1] - net[act,act - 8] + net[act,act - 9] 

  return np.sum(1.-net2)/64

# Analytical BER evaluation for 64QAM
def eval_analytic_BER_64QAM(SNR):
  sigma2w = SNR2sigma2w(SNR)
  QAM64 = np.asarray([i + 1j * j for i in np.arange(-7,0,2) for j in np.arange(-7,0,2)])/np.sqrt(42)
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in QAM64])
  x_shape = np.hstack([np.asarray([i for i in np.arange(-6,1,2)])/ np.sqrt(42)])
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], u, SI_c[i,1], sigma2w) \
  for t in x_shape for u in x_shape]).flatten() for i in range(0,16)]) 
  # p(x|s), where x are received const points and s are tranmitted const points
#  net2 = np.zeros(16)
  net2 = np.zeros(16)
  net2[0] = net[0,0] # p0
  net2[1] = net[1,1]-net[1,0] # p1
  net2[2] = net[2,2]-net[2,1] # p2
  net2[3] = net[3,3]-net[3,2] # p3

  for i in range(0,3):
    act = (i * 4) + 4 # act -4 -> down, act -1 -> left, act - 5 -> left down
    net2[act] = net[act, act] - net[act, act - 4] 
    act += 1
    for i in range(0,3):
      net2[act] = net[act, act] - net[act, act - 1] - net[act,act - 4] + net[act,act - 5] 
      act += 1

  return np.sum(1.-net2)/16

# Analytical BER evaluation for 32QAM
def eval_analytic_BER_32QAM(SNR):
  sigma2w = SNR2sigma2w(SNR)
# QAM32 = np.asarray([i + 1j * j for i in np.arange(-7,9,2) for j in np.arange(-3,5,2)]) / np.sqrt(26.)
  QAM32 = np.asarray([i + 1j * j for i in np.arange(-7,0,2) for j in np.arange(-3,0,2)])/np.sqrt(26)
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in QAM32])
  x_shape = np.hstack([np.asarray([i for i in np.arange(-6,1,2)])/ np.sqrt(26)])
  y_shape = np.hstack([np.asarray([i for i in np.arange(-2,1,2)])/ np.sqrt(26)])
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], u, SI_c[i,1], sigma2w) \
  for t in x_shape for u in y_shape]).flatten() for i in range(0,8)])
  # p(x|s), where x are received const points and s are tranmitted const points
  net2 = np.zeros(8)
  net2[0] = net[0,0] # p0
  net2[1] = net[1,1]-net[1,0] # p1
  net2[2] = net[2,2]-net[2,0] # p2
  net2[3] = net[3,3]-net[3,2]-net[3,1]+net[3,0] # p3
  net2[4] = net[4,4]-net[4,2] # p4
  net2[5] = net[5,5]-net[5,4] - net[5,3] + net[5,2] # p5
  net2[6] = net[6,6]-net[6,4] # p6
  net2[7] = net[7,7]-net[7,5]-net[7,6]+net[7,4] # p7

  return np.sum(1.-net2)/8

# Analytical BER evaluation for General QAM for even number of bits (2*N bits) and given mean energy per symbol
# N = 1 -> 4QAM, N = 2 -> 16QAM, N = 3 -> 64QAM, ...
def eval_analytic_BER_QAM_bits_even(SNR, N, Es = 1):
  sigma2w = SNR2sigma2w(SNR)
# QAM256 = np.asarray([i + 1j * j for i in np.arange(-15,17,2) for j in np.arange(-15,17,2)])/np.sqrt(170)
  QAMu = np.asarray([i + 1j * j  for i in np.arange(-2**N+1,0,2) for j in np.arange(-2**N+1,0,2)])
  en = (np.abs(QAMu)**2).sum() * 4 # overall QAM energy
  mult = np.sqrt(2.**(2*N) / en * Es)
  QAM = mult * QAMu
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in QAM])
  x_shape = np.hstack([np.asarray([i for i in np.arange(-2**N+2,1,2)]) * mult])
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], u, SI_c[i,1], sigma2w) \
  for t in x_shape for u in x_shape]).flatten() for i in range(0,2**(2*N-2))]) 
  # p(x|s), where x are received const points and s are tranmitted const points
#  net2 = np.zeros(16)
  net2 = np.zeros(2**(2*N-2))
  net2[0] = net[0,0] # p0
  for i in range(1,2**(N-1)):
    net2[i] = net[i, i] - net[i, i - 1]

  for i in range(1,2**(N-1)):
    act = i  * 2**(N-1)
    net2[act] = net[act, act] - net[act, act - 2**(N-1)] 
    for j in range(1,2**(N-1)):
      act = i * 2**(N-1) + j
      net2[act] = net[act, act] - net[act, act - 1] - net[act,act - 2**(N-1)] + \
      net[act,act - 2**(N-1) - 1] 

  return np.sum(1.-net2)/2**(2*N-2)

# Analytical BER evaluation for General QAM for odd number of bits (2*N + 1 bits) and given mean energy per symbol
# N = 1 -> 8QAM, N = 2 -> 32QAM, N = 3 -> 128QAM, ...
def eval_analytic_BER_QAM_bits_odd(SNR, N, Es = 1):
  sigma2w = SNR2sigma2w(SNR)
  QAMu = np.asarray([i + 1j * j  for i in np.arange(-2**(N+1)+1,0,2) for j in np.arange(-2**N+1,0,2)])
  en = (np.abs(QAMu)**2).sum() * 4 # overall QAM energy
  mult = np.sqrt(2.**(2*N + 1) / en * Es)
  QAM = mult * QAMu
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in QAM])
  x_shape = np.asarray([i for i in np.arange(-2**(N+1)+2,1,2)]) * mult
  y_shape = np.asarray([i for i in np.arange(-2**N+2,1,2)]) * mult
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], u, SI_c[i,1], sigma2w) \
  for t in x_shape for u in y_shape]).flatten() for i in range(0,2**(2*N-1))]) 
  # p(x|s), where x are received const points and s are tranmitted const points
#  net2 = np.zeros(16)
  net2 = np.zeros(2**(2*N-1))
  net2[0] = net[0,0] # p0
  for i in range(1,2**(N-1)): # loop up
    net2[i] = net[i, i] - net[i, i - 1]

  for i in range(1,2**N): # loop right
    act = i  * 2**(N-1)
    net2[act] = net[act, act] - net[act, act - 2**(N-1)] 
    for j in range(1,2**(N-1)): # loop up
      act = i * 2**(N-1) + j
      net2[act] = net[act, act] - net[act, act - 1] - net[act,act - 2**(N-1)] + \
      net[act,act - 2**(N-1) - 1] 

  return np.sum(1.-net2)/(2**(2*N - 1))


def eval_analytic_BER_HDF8ex(SNR):
  sigma2w = SNR2sigma2w(SNR)
  QAM16 = np.asarray([i + 1j * j for i in np.arange(-3,5,2) for j in np.arange(-3,5,2)]) /np.sqrt(5)
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in QAM16])
  x_shape = np.hstack([np.asarray([i for i in np.arange(-2,3,2)])/ np.sqrt(5), np.inf])
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], u, SI_c[i,1], sigma2w) \
  for t in x_shape for u in x_shape]).flatten() for i in range(0,16)]) 
  # p(x|s), where x are received const points and s are tranmitted const points
#  net2 = np.zeros(16)
  net2 = np.zeros(4)
  net2[0] = net[0,0] # p0
  net2[1] = net[1,1]-net[1,0] # p1
#  net2[2] = net[2,2]-net[2,1] # p2
#  net2[3] = net[3,3]-net[3,2] # p3

  net2[2] = net[4,4] - net[4,0] # p4
  net2[3] = net[5,5]-net[5,4] - net[5, 1] + net[5, 0] # p5
#  net2[4] = net[4,4] # p4
#  net2[5] = net[5,5]-net[5,4] - net[5, 1] + net[5, 0] # p5
#  net2[6] = net[6,6]-net[6,5] - net[6, 2] + net[6, 1] # p6
#  net2[7] = net[7,7]-net[7,6] - net[7, 3] + net[7, 1] # p7

#  for i in range(0,3):
#    net2[(i * 4) +4] = net[(i * 4) +4,(i * 4) +4] - net[(i * 4) +4,(i * 4)]    # p(i * 4) +4

#    net2[(i * 4) +5] = net[(i * 4) +5,(i * 4) +5] - net[(i * 4) +5,(i * 4) +4] -\
#                       net[(i * 4) +5,(i * 4) +1] + net[(i * 4) +5,(i * 4)]    # p(i * 4) +5

#    net2[(i * 4) +6] = net[(i * 4) +6,(i * 4) +6] - net[(i * 4) +6,(i * 4) +5] -\
#                       net[(i * 4) +6,(i * 4) +2] + net[(i * 4) +6,(i * 4) +1] # p(i * 4) +6

#    net2[(i * 4) +7] = net[(i * 4) +7,(i * 4) +7] - net[(i * 4) +7,(i * 4) +6] -\
#                       net[(i * 4) +7,(i * 4) +3] + net[(i * 4) +7,(i * 4) +2] # p(i * 4) +7

  return np.sum(1.-net2)/4

# Analytic exteded HDF site-link
def eval_analytic_BER_site_link(SNR):
  BPSK = np.exp(1j * np.arange(0,2) * np.pi) 
  alph = 0.2
  Ba = np.sqrt(1 - alph) #   Basic part --- dA2, dB2 
  Su = np.sqrt(alph) #  Superposed part --- dA1, dB1  
  sigma2w = SNR2sigma2w(SNR)
  Pbe = 1 - (prod_gauss2D(0, -Ba,np.inf , Su,sigma2w) + prod_gauss2D(0, -Ba,np.inf , -Su,sigma2w))/2
  return Pbe

# Analytic exteded HDF site-link (1 bit site link and 3 bits to relay)
def eval_analytic_BER_site_link_1bit(SNR):
  alph = 1./21
  Ba = 4 * np.sqrt(alph) #   Basic part 
  sigma2w = SNR2sigma2w(SNR)
  Pbe = 1 - 0.5 * (1. + erf((Ba) / np.sqrt(sigma2w)))
  return Pbe

# Analytic exteded HDF site-link (2 bits site link and 3 bits to relay) 
def eval_analytic_BER_site_link_2bits(SNR):
  BPSK = np.exp(1j * np.arange(0,2) * np.pi) 
  alph = 1./21
  Ba = 4 * np.sqrt(alph) + 2j * np.sqrt(alph) #   Basic part 
  sigma2w = SNR2sigma2w(SNR)
  Pbe = 1 - prod_gauss2D(0, -np.real(Ba), 0 , -np.imag(Ba),sigma2w)
  return Pbe


## Analytical BER Evaluations for the proposed constellation designs

# A generic Pbe evaluation for constellation with square lattice with multiple observ
def eval_analytic_square_grid(SNR, const):
#  Constellation taking into the account only the second quadrant
  sigma2w = SNR2sigma2w(SNR)
  constr = np.sort_complex(const)
  constr_un = np.unique(np.round(constr,10))
  re_vals = np.unique(np.real(constr_un))
  im_vals = np.unique(np.imag(constr_un))
#  re_vals = np.unique(np.sort(np.real(np.round(constr,6))))
#  im_vals = np.unique(np.sort(np.imag(np.round(constr,6))))
  re_borders = np.hstack([np.diff(re_vals)/2 + re_vals[:-1], np.inf])
  im_borders = np.hstack([np.diff(im_vals)/2 + im_vals[:-1], np.inf])
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in constr])
  SI_c_un = np.asarray([(np.real(i), np.imag(i)) for i in constr_un])
  x_shape = re_borders
  y_shape = im_borders
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c_un[i,0], u, SI_c_un[i,1], sigma2w) \
  for t in x_shape for u in y_shape]).flatten() for i in range(0,len(constr_un))]) 
  # p(x|s), where x are received const points and s are tranmitted const points
#  net2 = np.zeros(16)
  hist = np.asarray([[((np.round(SI_c,10)[:,0] == re_vals[i]) * (np.round(SI_c,10)[:,1] == im_vals[j])).sum() \
         for i in range(0,len(re_vals))] for j in range(0,len(im_vals))])
  net2 = np.zeros(len(hist.flatten()))
  net2[0] = net[0,0] # p0

  for i in range(1,len(im_borders)): # loop up
    net2[i] = (net[i, i] - net[i, i - 1])
  for i in range(1,len(re_borders)): # loop right
    act = i  * len(im_borders)
    net2[act] = (net[act, act] - net[act, act - len(im_borders)])
    for j in range(1,len(im_borders)): # loop up
      act = i * len(im_borders) + j
      net2[act] = (net[act, act] - net[act, act - 1] - net[act,act - len(im_borders)] + \
      net[act,act - len(im_borders) - 1])
  Pe = 0.
  for i in range(0,len(im_borders)):
    for j in range(0,len(re_borders)):
      act = i * len(im_borders) + j
      Pe += hist[i, j] * (1. - net2[act])

  return Pe/len(const)


# A generic Pbe evaluation for constellation with square lattice that is symmetric around zero
def eval_analytic_square_grid_const(SNR, const):
#  Constellation taking into the account only the second quadrant
  sigma2w = SNR2sigma2w(SNR)
  quadr2t = const[np.intersect1d(np.nonzero(np.real(const) <= 0)[0],np.nonzero(np.imag(const) <= 0)[0])]
  quadr2  = np.sort_complex(quadr2t)
  re_vals = np.unique(np.sort(np.real(quadr2)))
  im_vals = np.unique(np.sort(np.imag(quadr2)))
  re_borders = np.hstack([np.diff(re_vals)/2 + re_vals[:-1], 0])
  im_borders = np.hstack([np.diff(im_vals)/2 + im_vals[:-1], 0])
  SI_c = np.asarray([(np.real(i), np.imag(i)) for i in quadr2])
  x_shape = re_borders
  y_shape = im_borders
  net = np.asarray([np.asarray([prod_gauss2D(t, SI_c[i,0], u, SI_c[i,1], sigma2w) \
  for t in x_shape for u in y_shape]).flatten() for i in range(0,len(quadr2))]) 
  # p(x|s), where x are received const points and s are tranmitted const points
#  net2 = np.zeros(16)
  net2 = np.zeros(len(quadr2))
  net2[0] = net[0,0] # p0
  for i in range(1,len(im_vals)): # loop up
    net2[i] = net[i, i] - net[i, i - 1]
  for i in range(1,len(re_vals)): # loop right
    act = i  * len(im_vals)
    net2[act] = net[act, act] - net[act, act - len(im_vals)] 
    for j in range(1,len(im_vals)): # loop up
      act = i * len(im_vals) + j
      net2[act] = net[act, act] - net[act, act - 1] - net[act,act - len(im_vals)] + \
      net[act,act - len(im_vals) - 1] 
  return np.sum(1.-net2)/len(net2)

def eval_analytic_BER_JDF_superposed(Nb, Ns, SNR):
  return eval_analytic_BER_QAM_bits_even(SNR, (Nb + Ns), 2)

def eval_analytic_BER_JDF_basic(Nb, Ns, SNR):
  const = JDF_consts_design_only_basic(Nb, Ns) # Basic part of const
  if (Nb == 1):
    sigma2w = SNR2sigma2w(SNR)
    return 1- 0.5 * (1. + erf(np.abs(const[0]) / np.sqrt(sigma2w)))
  return eval_analytic_square_grid_const(SNR, const)

def eval_analytic_BER_HDF_basic(Nb, Ns, SNR):
  const = HDF_consts_design_basic(Nb, Ns) 
  if (Nb == 1):
    sigma2w = SNR2sigma2w(SNR)
    return 1- 0.5 * (1. + erf(np.abs(const[0]) / np.sqrt(sigma2w)))
  else:
    return eval_analytic_square_grid_const(SNR, const)

def eval_analytic_BER_HDF_superposed(Nb, Ns, SNR):
  const = HDF_consts_design(Nb, Ns)
  return eval_analytic_square_grid(SNR, const)
