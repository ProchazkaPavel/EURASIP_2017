import numpy as np
from analytical_BER import SNR2sigma2w
from Block_functions import Error_Check_Relay, Demodulator_relay, Modulator_network 

def AWGN(signal, SNR): # Unit power per dimension assumed
  sigma2w = SNR2sigma2w(SNR)
  real_noise = np.random.normal(0,np.sqrt(sigma2w),np.shape(signal/2))
  imag_noise = 1j * np.random.normal(0,np.sqrt(sigma2w),np.shape(signal/2))
  noise = 1.0/np.sqrt(2)  * (real_noise + imag_noise)
  return signal + noise

### NUMERICAL PART (BER SIMULATION)

# Numerical BER evaluation for QPSK
def eval_numerical_BER_QPSK(SNR_vals, N):
  d = np.random.randint(4,size=N)
  QPSK = np.exp(1j * (np.pi * np.arange(0,4)/2.0 + np.pi/4))
  s = QPSK[d]
  BER = []
  for SNR in SNR_vals:
    est_d = np.zeros(N)
    x = AWGN(s, SNR)
    est_d[np.intersect1d(np.nonzero(np.real(x) > 0)[0],np.nonzero(np.imag(x) > 0)[0])] = 0
    est_d[np.intersect1d(np.nonzero(np.real(x) < 0)[0],np.nonzero(np.imag(x) > 0)[0])] = 1
    est_d[np.intersect1d(np.nonzero(np.real(x) < 0)[0],np.nonzero(np.imag(x) < 0)[0])] = 2
    est_d[np.intersect1d(np.nonzero(np.real(x) > 0)[0],np.nonzero(np.imag(x) < 0)[0])] = 3
    BER.append(np.sum(est_d != d))
  return BER

# Numerical BER evaluation for 8QAM
def eval_numerical_BER_8QAM(SNR_vals, N):
  d = np.random.randint(8,size=N)
  QAM8 = np.asarray([i + 1j * j for i in np.arange(-3,5,2) for j in np.arange(-1,3,2)]) /np.sqrt(6)
  s = QAM8[d]
  BER = []
  for SNR in SNR_vals:
    est_d = np.zeros(N)
    x = AWGN(s, SNR)
    mu = np.asarray([np.abs(x - s0)**2 for s0 in QAM8])
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical BER evaluation for 16QAM
def eval_numerical_BER_16QAM(SNR_vals, N):
  d = np.random.randint(16,size=N)
  QAM16 = np.asarray([i + 1j * j for i in np.arange(-3,5,2) for j in np.arange(-3,5,2)]) /  np.sqrt(10)
  s = QAM16[d]
  BER = []
  for SNR in SNR_vals:
    est_d = np.zeros(N)
    x = AWGN(s, SNR)
    mu = np.asarray([np.abs(x - s0)**2 for s0 in QAM16])
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical BER evaluation for 64QAM
def eval_numerical_BER_64QAM(SNR_vals, N):
  d = np.random.randint(64,size=N)
  QAM64 = np.asarray([i + 1j * j for i in np.arange(-7,9,2) for j in np.arange(-7,9,2)])/np.sqrt(42)
  s = QAM64[d]
  BER = []
  for SNR in SNR_vals:
    est_d = np.zeros(N)
    x = AWGN(s, SNR)
    mu = np.asarray([np.abs(x - s0)**2 for s0 in QAM64])
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

def eval_numerical_BER_256QAM(SNR_vals, N):
  d = np.random.randint(256,size=N)
  QAM256 = np.asarray([i + 1j * j for i in np.arange(-15,17,2) for j in np.arange(-15,17,2)])/np.sqrt(170)
  s = QAM256[d]
  BER = []
  for SNR in SNR_vals:
    est_d = np.zeros(N)
    x = AWGN(s, SNR)
    mu = np.asarray([np.abs(x - s0)**2 for s0 in QAM256])    
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical BER evaluation for QPSK with HDF
def eval_numerical_BER_QPSK_HDF(SNR_vals, N):
  dA = np.random.randint(4,size=N)
  dB = np.random.randint(4,size=N)
  QPSK = np.exp(1j * (np.pi * np.arange(0,4)/2.0 + np.pi/4))
  const_superposed = np.round(np.asarray([i + j for i in QPSK for j in QPSK]).flatten(),8)
  s = QPSK[dA] + QPSK[dB]
  BER = []
  for SNR in SNR_vals:
    est_d = np.zeros(N)
    x = AWGN(s, SNR)
    mu = np.zeros([16, N]) # Joint metric
    muM = np.zeros([4,N]) # Min hier. metric
    sigma2w = SNR2sigma2w(SNR)

    mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in const_superposed])
    for i in range(0, len(const_superposed)):
      vec = index2vec(i, np.ones(4,int))
      j = (vec[0] ^ vec[2]) * 2 + vec[1]^vec[3]
      muM[j, :] += mu[i, :]
    est_d = muM.argmax(axis=0)

    BER.append(np.sum(est_d != (dA ^ dB)))
#    BER.append((np.abs(est_d - d) > 1e-8).sum())
  return BER

# Numerical BER evaluation for 8QAM
def eval_numerical_BER_8QAM(SNR_vals, N):
  d = np.random.randint(8,size=N)
  QAM8 = np.asarray([i + 1j * j for i in np.arange(-1,3,2) for j in np.arange(-3,5,2)])/np.sqrt(6)
  s = QAM8[d]
  BER = []
  for SNR in SNR_vals:
    x = AWGN(s, SNR)
    mu = np.asarray([np.abs(x - s0)**2 for s0 in QAM8])
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical BER evaluation for 8PSK
def eval_numerical_BER_8PSK(SNR_vals, N):
  d = np.random.randint(8,size=N)
  PSK8 = np.exp(1j * (np.pi * np.arange(0,8)/4.0))
  s = PSK8[d]
  BER = []
  for SNR in SNR_vals:
    x = AWGN(s, SNR)
    mu = np.asarray([np.abs(x - s0)**2 for s0 in PSK8])
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical BER evaluation for extended cardinality in relay
def eval_numerical_BER_exHDF(SNR_vals, N):
  BPSK = np.exp(1j * np.arange(0,2) * np.pi) 
  alph = 0.2
  Ba = np.sqrt(1 - alph) #   Basic part --- dA2, dB2 
  Su = np.sqrt(alph) #  Superposed part --- dA1, dB1  
  dA1 = np.random.randint(2,size=N)
  dA2 = np.random.randint(2,size=N)
  dB1 = np.random.randint(2,size=N)
  dB2 = np.random.randint(2,size=N)
  d = dA1 * 4 +dB1 * 2 + (dA2^dB2)
  # Source processing
  sA = Su * BPSK[dA1] + 1j * Ba * BPSK[dA2]
  sB = Su * BPSK[dB1] * 1j + Ba * BPSK[dB2]
  BER = []
  for SNR in SNR_vals:
    x = AWGN(sA + sB, SNR) # signal received in relay
    sigma2w = SNR2sigma2w(SNR)
    conR = np.asarray([Su * BPSK[i] + 1j * Ba * BPSK[j] + Su * BPSK[k] * 1j + Ba * BPSK[l] \
		for i in range(0,2) for j in range(0,2) for k in range(0,2) for l in range(0,2)])

    mu = np.zeros([16, N]) # Joint metric
    muH = np.zeros([8,N]) # Extended hier. metric
    # Metric evaluation in relay
    mu = np.asarray([np.exp(-np.abs(x - s0)**2/sigma2w) for s0 in conR])
    for i in range(0, len(conR)):
      vec = index2vec(i, np.ones(4,int))
      j = 4 * vec[0] + 2 * vec[2] + (vec[1] ^ vec[3]) 
      muH[j, :] += mu[i, :]
    est_d = muH.argmax(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical BER evaluation for full cardinality in relay
def eval_numerical_BER_fullHDF(SNR_vals, N):
  d = np.random.randint(16,size=N)
  BPSK = np.exp(1j * np.arange(0,2) * np.pi) 
  alph = 0.2
  Ba = np.sqrt(1 - alph) #   Basic part --- dA2, dB2 
  Su = np.sqrt(alph) #  Superposed part --- dA1, dB1  
  conR = np.asarray([Su * BPSK[i] + 1j * Ba * BPSK[j] + Su * BPSK[k] * 1j + Ba * BPSK[l] \
         for i in range(0,2) for j in range(0,2) for k in range(0,2) for l in range(0,2)])
  s = conR[d]
  BER = []
  for SNR in SNR_vals:
    est_d = np.zeros(N)
    x = AWGN(s, SNR)
    mu = np.asarray([np.abs(x - s0)**2 for s0 in conR])
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical H-SI BER evaluation for extended cardinality 
def eval_numerical_BER_site_link(SNR_vals, N):
  alph = 0.2
  Ba = np.sqrt(1 - alph) #   Basic part --- dA2, dB2 
  BPSK = np.exp(1j * np.arange(0,2) * np.pi)  * Ba
  d = np.random.randint(2,size=N)
  s = BPSK[d]
  BER = []
  for SNR in SNR_vals:
    x = AWGN(s, SNR) # signal received in relay
    mu = np.asarray([np.abs(x - s0)**2 for s0 in BPSK])
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical H-SI BER evaluation for extended cardinality  (1 of 3bits)
def eval_numerical_BER_site_link_1bit(SNR_vals, N):
  alph = 1./21
  Ba = 4 * np.sqrt(alph) # 
  BPSK = np.exp(1j * np.arange(0,2) * np.pi)  * Ba
  d = np.random.randint(2,size=N)
  s = BPSK[d]
  BER = []
  for SNR in SNR_vals:
    x = AWGN(s, SNR) # signal received in relay
    mu = np.asarray([np.abs(x - s0)**2 for s0 in BPSK])
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical H-SI BER evaluation for extended cardinality  (2 of 3bits)
def eval_numerical_BER_site_link_2bits(SNR_vals, N):
  alph = 1./21
  Ba1 = 4 * np.sqrt(alph) # 
  Ba2 = 2j * np.sqrt(alph) # 
  Ba3 = np.sqrt(alph)
  BPSK = [-1.,1.]
  const = np.asarray([a * Ba1  + b * Ba2 + c * Ba3 \
                      for a in BPSK for b in BPSK for c in BPSK])
  d = np.random.randint(4,size=N)
  s = const[d]
  BER = []
  for SNR in SNR_vals:
    x = AWGN(s, SNR) # signal received in relay
    mu = np.asarray([np.abs(x - s0)**2 for s0 in const])
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical BER simulation for general constellation
def eval_numerical_BER_general_mod(SNR_vals, const, N):
  d = np.random.randint(len(const),size=N)
  s = const[d]
  BER = []
  for SNR in SNR_vals:
    est_d = np.zeros(N)
    x = AWGN(s, SNR)
    mu = np.asarray([np.abs(x - s0)**2 for s0 in const])    
    est_d = mu.argmin(axis=0)
    BER.append(np.sum(est_d != d))
  return BER

# Numerical BER simulation 
def eval_numerical_BER_HDF_gen_MAC(SNR, Nb, Ns, N, mapping = 'XOR', h = 1):
  sigma2w = SNR2sigma2w(SNR)
  # Source A
  dA = np.random.randint(2**Nb * 2**Ns, size = N)
  sA = Modulator_network(dA, Nb, Ns, 1, mapping)
  
  # Source B
  dB = np.random.randint(2**Nb * 2**Ns, size = N)
  sB = Modulator_network(dB, Nb, Ns, 2, mapping)
  
  # MAC channel
  w_MAC = np.sqrt(sigma2w/2)*(np.random.normal(size=N) + 1j*np.random.normal(size=N))
  x = sA + h * sB + w_MAC
  
  #plt.plot(numpy.real(x), numpy.imag(x),'xk',ms=5)
  #plt.plot(numpy.real(sA), numpy.imag(sA),'or',ms=5)
  #plt.plot(numpy.real(sB), numpy.imag(sB),'db',ms=5)
  #plt.show()
  
  # Relay Processing
  dH = Demodulator_relay(x, Nb, Ns, mapping, 1, h)
  return Error_Check_Relay(dH, dA, dB, Nb, Ns, mapping)

# Numerical BER simulation for HDF mixed --- NOT FINISHED
def eval_numerical_BER_HDF_gen_MAC_m(SNR_vals, Nb, Ns, N):
#  constA = HDF_consts_design_one_user(Nb, Ns)
#  constB = 1j * HDF_consts_design_one_user(Nb, Ns)
  const = HDF_consts_design(Nb, Ns)
  dAb = np.random.randint(2**Nb, size=N) # basic part A
  dAs = np.random.randint(2**Ns, size=N) # superposed part A
  dBb = np.random.randint(2**Nb, size=N) # basic part B
  dBs = np.random.randint(2**Ns, size=N) # superposed part B
#  sA = constA[dAb + 2**Nb * dAs]
#  sB = constB[dBb + 2**Nb * dBs]
  s = const[dBb + 2**Nb * dBs + 2**(Ns+Nb) * dAb + 2**(2*Nb+Ns) * dAs]
  BER = []
  for SNR in SNR_vals:
    est_d = np.zeros(N)
    x = AWGN(s, SNR)
    mu = np.asarray([np.abs(x - s0)**2 for s0 in const])    
    est_r = mu.argmin(axis=0)
    bin_note = dec2bin(est_r,2*Nb+2*Ns)
    est_d = bin2dec(bin_note[0:Nb] ^ bin_note[Nb+Ns:2*Nb+Ns])
    BER.append(np.sum(est_d != (dAb ^ dBb)))
  return np.asarray(BER)/float(N)
