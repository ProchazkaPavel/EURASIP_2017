import os, sys, inspect
sys.path.insert(0,'lib')
from rew import *
import LDPC_lib 
import GFq_LDPC_lib
from GFq_LDPC_lib import  *
from compare_QPSK import QAM,const_design_XOR, const_superposed
from Turbo_lib import *


import _Turbo_lib
import _GFq_LDPC_lib
import _LDPC_dec_lib

def const_map(bits):
  '''
  Conversion of N-bit binary data streams - (n * M x N) array to integer prepared for modulation.

  Parameters:
  -----------
  M : int
      Number of frames.
  n : int
      Length of frame.
  N : int
      Number of bits in constellation.
  bits : array of ints
      (n * M x N) 2D numpy array with binary data.

  Returns: 
  --------
  inds : array of ints
       (M * n) 1D numpy array with integer prepared to be modulated

  See Also:
  ---------
  soft_demap

  Examples:
  ---------
  
  '''
  [x, N] = np.shape(bits)
  res = np.zeros(x, int)
  for i in range(N):
    res += bits[:, i] * (2 ** (N - i - 1))

  return res
  
def demod(x, const, sigma2w, h = 1, norming = 0):
  '''
  Demodulation related to a given N-bit modulator. Assuming a P2P channel with Gaussian noise with
  varianve sigma2w.

  Parameters:
  -----------
  x : array of complex
      Observation vector given by s + w, where s is modulated signal and w is iid. AWGN with variance
      sigma2w.
  const : array of complex
      The one (complex) dimensional signal space mapping such that s = const[c_int], where c_int is
      the index of carrying the information.
  sigma2w : complex
      Variance related to Gaussian noise in AWGN.
  h : complex or array of complex
      Fading coefficient. One can consider no fading (default value h = 1), block fading h = any
      complex number or fast fading, where h can be also a complex array with the sime size as x      
  norming : int
      Flag if the resulting message should be normalized to pmf (i.e. the sum of each line of
      resulting soft output should be 1)

  Returns: 
  --------
  mu : array of floats
       (M * n) x 2**N 2D numpy array, where 2**N is length of const. The matrix is soft metric for
       decodign. 


  See Also:
  ---------
  soft_demap, const_map, AWGN

  Examples:
  ---------
  import numpy as np
  n = 1000                                            # Length of the data message
  const = np.exp(1j * (2 * np.pi * np.arange(8)) / 8) # 8-PSK constellation
  SNR = 10                                            # Signal to noise ration in dB
  c = np.random.randint(8, size = n)                  # N random integers
  s = const[c]                                        # Signal space mapping
  (x,sigma2w) = AWGN(s, SNR)                          # AWGN channel
  mu = demod(x, const, sigma2w, norming = 0)           
  est_c = np.argmax(mu, axis=1)                       # estimation


  mu = demod(x, const, sigma2w, norming = 1)           
  np.sum(mu, axis = 1)                                # verify the normalization


  '''

  lenX, = x.shape
  lenC = len(const)
  mu = np.zeros([lenX, lenC], float) 

  for i in range(0, lenC):
    mu[:, i] = np.exp(-np.abs(x - h * const[i])**2/sigma2w)

  if (np.min(mu) == 0):
    mu += 1e-100   # Awoiding all-zero vector
 
  if norming:
    mu = mu / np.sum(mu,axis = 1).repeat(lenC).reshape(lenX, lenC)

  return mu


def soft_demap(muR, mu_o, n, M, N, norming = 0):
  '''
  Soft output demodulator. Supposing N-bit constellation demodulated to the soft metric given by 
  (n x 2**N) matrix muR. A priory metric from coder is in mu_o, which is a 3D (M * n x 2 x N) vector.

  Parameters:
  -----------
  M : int
      Number of frames.
  n : int
      Length of frame.
  N : int
      Number of bits in constellation.
  muR : array of floats
      (n x 2**N) 2D numpy array with input (observation) vector.
  mu_o : array of floats
      (M * n x 2 x N) 3D numpy array with a priory observation.
  norming : int
      Flag if the resulting message should be normalized to pmf (i.e. the sum of each line of
      resulting soft output should be 1)

  Returns: 
  --------
  mu : array of floats
       (M * n x 2 x N) 3D numpy array with updated soft information

  See Also:
  ---------
  const_map, demap, AWGN

  Examples:
  ---------
  import numpy as np
  n = 10                                                    # Length of the data message
  M = 1                                                     # Numer of frames
  N = 3                                                     # N-bit constellation
  const = np.exp(1j * (2 * np.pi * np.arange(2**N)) / 2**N) # N-PSK constellation
  SNR = 18                                                  # Signal to noise ration in dB
  b = np.random.randint(2, size = (n * M, N))               # N random integers
  c = const_map(b)                                          # Conversion of binary vector to integers
  s = const[c]                                              # Signal space mapping
  (x,sigma2w) = AWGN(s, SNR)                                # AWGN channel
  muR = demod(x, const, sigma2w, norming = 0)               # Soft output demodulator
  mu_o = np.ones([M * n, 2, N]) * 0.5                       # A priory information
  mu = soft_demap(muR, mu_o, n, M, N)                       # Update of soft message with a uniform priory info
  est_b = np.argmax(mu,axis=1)                              # Estimation of bits
  print 'nerr=', np.sum(b != est_b)                         # Verification
  '''

  mu = np.zeros([n * M, 2, N], float)
  for i in range(1 << N):
    inds = dec2bin(i, N)
    for j in range(N):
      temp = np.asarray([mu_o[:, inds[l], l] for l in range(N) if l != j])
      mu[:, inds[j], j] += muR[:, i] * temp.prod(axis = 0)
  return mu

def interleave2D(inter, matrix):
  '''
  Interleave a given matrix according to given interleaver

  Parameters:
  -----------
  matrix : array of ints
      n x k 2D numpy array to be inteleaved.
  inter : array of ints
      Interleaver.

  Returns: 
  --------
  out : array of ints
       (n x k) 2D numpy interleaved array. 

  See Also:
  ---------

  Examples:
  ---------
  n = 10
  k = 3
  inter = np.random.permutation(n*k)
  deinter = np.zeros(n*k, int)
  for i in range(0, n*k):                      # Interleaver creation
    deinter[inter[i]] = i
  
  b = np.random.randint(2,size=[n,k])          # Data creation
  b_int = interleave2D(inter, b)               # Interleaved data
  b_dei = interleave2D(deinter, b_int)         # Deinterleave data b_dei = b

  '''
  return matrix.flatten()[inter].reshape(matrix.shape)
 
def interleave_metric(inter, mu):
  '''
  Interleave (deinterleave) a given metric according to given interleaver in a single dimension.

  Parameters:
  -----------
  mu : array of floats
       (M * n x 2**N) 2D numpy array with soft information
  inter : array of ints
      Interleaver.

  Returns: 
  --------
  mu_i : array of ints
       corresponding (M * n x 2**N) 2D numpy interleaved array. 

  See Also:
  ---------
  iterleave2D, 

  Examples:
  ---------
  n = 50
  N = 3
  M = 10
  inter = np.random.permutation(n*M)
  deinter = np.zeros(n*M, int)
  for i in range(0, n*M):                                   # Interleaver creation
    deinter[inter[i]] = i
  
  const = np.exp(1j * (2 * np.pi * np.arange(2**N)) / 2**N) # N-PSK constellation
  SNR = 18                                                  # Signal to noise ration in dB

  b = np.random.randint(2**N, size = n * M)                 # Random data
  c = b[inter]                                              # Interleaved data
  s = const[c]                                              # Signal space mapping
  (x,sigma2w) = AWGN(s, SNR)                                # AWGN channel
  mu = demod(x, const, sigma2w, norming = 0)                # Soft output demodulator
  mu_d = interleave_metric(deinter, mu)                     # Deinterleave the metric
  est_b = np.argmax(mu_d,axis=1)                            # Estimation of bits
  print np.sum(est_b != b)
  ''' 
  return mu[inter]


def interleave2D_metric(inter, mu):
  '''
  Interleave (deinterleave) a given metric according to given interleaver in 2 dimensions

  Parameters:
  -----------
  mu : array of floats
       (M * n x 2 x N) 3D numpy array with updated soft information
  inter : array of ints
      Interleaver.

  Returns: 
  --------
  mu_i : array of ints
       corresponding (M * n x 2 x N) 3D numpy interleaved array. 

  See Also:
  ---------
  iterleave2D

  Examples:
  ---------
  n = 50
  N = 3
  M = 10
  inter = np.random.permutation(n*N*M)
  deinter = np.zeros(n*N*M, int)
  for i in range(0, n*N*M):                                 # Interleaver creation
    deinter[inter[i]] = i
  
  const = np.exp(1j * (2 * np.pi * np.arange(2**N)) / 2**N) # N-PSK constellation
  SNR = 18                                                  # Signal to noise ration in dB

  b = np.random.randint(2, size = (n * M, N))               # N random integers
  b_int = interleave2D(inter, b)                            # Interleaved data
  c = const_map(b_int)                                          # Conversion of binary vector to integers
  s = const[c]                                              # Signal space mapping
  (x,sigma2w) = AWGN(s, SNR)                                # AWGN channel
  muR = demod(x, const, sigma2w, norming = 0)               # Soft output demodulator
  mu_o = np.ones([M * n, 2, N]) * 0.5                       # A priory information
  mu = soft_demap(muR, mu_o, n, M, N)                       # Update of soft message with a uniform priory info
  mu_d = interleave2D_metric(deinter, mu)                   # Deinterleave at the metric level
  est_b = np.argmax(mu_d,axis=1)                            # Estimation of bits

  '''
  [x, tt, N] = mu.shape
  mu_i = np.zeros([x, 2, N], float)
  mu_i[:,0,:] = interleave2D(inter, mu[:,0,:])
  mu_i[:,1,:] = interleave2D(inter, mu[:,1,:])
  return mu_i
  


def HDF_MAC_Signal_Space(Nb, Ns, SNR, cAs_int, cAb_int, cBs_int, cBb_int, hA = 1, hB = 1): # Binnary
  '''
  Overall function for MAC channel with HDF that can be run either as numerical verification of
  uncoded system or to be a   part of coded system caring about signal space transmission starting
  from integers to be carried   and ended by soft hierarchica metric evaluated in relay.

  Parameters:
  -----------
  Nb : int
      Number of basic bits (suited to be decoded as XOR function in relay) 
  Ns : int
      Number of superposed bits (suited to be decoded separately in relay) 
  SNR : float
      Signal to noise ratio.
  cAs_int : array of ints
      Information integers referring to superposed part in source A
  cAb_int : array of ints
      Information integers referring to basic part in source A
  cBs_int : array of ints
      Information integers referring to superposed part in source B
  cBb_int : array of ints
      Information integers referring to basic part in source B
  hA, hB : complex or arrays of complex
      Fading coefficients. One can consider no fading (default value hA = 1, hB = 1), block fading
      hA, hB = any complex number or fast fading, where (hA, hB) can be also a complex arrays with
      the sime size as x. Note that the relay observation is given by x = hA * sA + hB * sB + w,
      where sA, sB are signals in sources A, B and w is AWGN.

  Returns: 
  --------
  muH : array of floats
       corresponding to (M * n x 2**(Nb + 2* Ns) 2D numpy array referring to hierarchical metric
       evaluated in relay. 

  See Also:
  ---------

  Examples:
  ---------
  HDF_MAC(2, 2, 1000, 20, 2) 
  '''
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h = 1)

  cR_int = (cAs_int << (Nb+Ns)) + (cBs_int << Nb) + cAb_int ^ cBb_int
  cA_int = (cAs_int << Nb) + cAb_int
  cB_int = (cBs_int << Nb) + cBb_int

  sA = sourceA_const[cA_int]
  sB = sourceB_const[cB_int]

  (x,sigma2w) = AWGN(hA*sA + hB*sB, SNR)

  N = len(cA_int)

  muR = np.zeros([N * M, 2 ** (2*Nb + 2*Ns)], float)
  muH = np.zeros([N * M, 2 ** (Nb + 2*Ns)], float) 

  for iA in range(0, 1 << (Nb + Ns)):
    for iB in range(0, 1 << (Nb + Ns)):
      ind = (iA << (Nb + Ns)) + iB
      muR[:, ind] = np.exp(-np.abs(x - hA * sourceA_const[iA] - hB * sourceB_const[iB])**2/sigma2w) 
      inds = dec2bin(ind, 2*Nb + 2*Ns)
      if Ns == 0: # min map
        hier_ind = bin2dec(np.asarray([inds[j] ^ inds[j+Nb] for j in range(Nb)]))
      elif Nb == 0: # JDF
        hier_ind = ind                                     
      else:
        hier_ind = bin2dec(np.hstack([[inds[j] for j in range(Ns)],\
                                     [inds[j] for j in range(Nb+Ns,Nb+2*Ns)],\
                                     [inds[j] ^ inds[j+Nb+Ns] for j in range(Ns,Nb+Ns)]]))
      muH[:, hier_ind] += muR[:, ind]    

  if (np.min(muH) == 0):
    muH += 1e-100
#  print 'Uncoded symbol errors', np.sum(np.argmax(muH, axis = 1) != cR_int)
  return muH

def test_MAC_bits(Nb, Ns, N, SNR, M, hA = 1, hB = 1):
  cAs = np.random.randint(2, size=(M * N, Ns))
  cBs = np.random.randint(2, size=(M * N, Ns))
  cAb = np.random.randint(2, size=(M * N, Nb))
  cBb = np.random.randint(2, size=(M * N, Nb))

  cAs_int = const_map(cAs) 
  cBs_int = const_map(cBs) 
  cAb_int = const_map(cAb) 
  cBb_int = const_map(cBb) 

  muH =  HDF_MAC_Signal_Space(Nb, Ns, SNR, cAs_int, cAb_int, cBs_int, cBb_int)

  mu_o = np.ones([M * N, 2, Nb + 2*Ns]) * 0.5                       # A priory information
  mu = soft_demap(muH, mu_o, N, M, Nb + 2*Ns, norming = 1)
  
  est_c = np.argmax(mu, axis=1)  
  print 'Uncoded bit errors', np.sum(est_c != np.hstack([cAs, cBs, cAb ^ cBb]))

def create_code(Aq = 2, depth = 3):
  '''
  Aq-ary alphabet with given depth. Number of modulator states (and corresponding complexity) is
  given by Aq ** (depth)
  '''
  if depth == 3:
    S = np.zeros([Aq,Aq**3],int)
    Q = np.zeros([Aq,Aq**3], int)
    for s in range(Aq**3):
      for d in range(Aq):
        s0 = s % Aq
        s1 = (s / Aq) % Aq
        s2 = (s / Aq**2) % Aq
        S[d,s] = Aq**2 * (s2 ^ d) + Aq * (s0 ^ s2) + s1
        Q[d,s] = s2 ^ s0 ^ d

  elif depth == 4:
    S = np.zeros([Aq,Aq**4],int)
    Q = np.zeros([Aq,Aq**4], int)
    for s in range(Aq**4):
      for d in range(Aq):
        s0 = s % Aq
        s1 = (s / Aq) % Aq
        s2 = (s / Aq**2) % Aq
        s3 = (s / Aq**3) % Aq
        S[d,s] = Aq**3 * (s3 ^ d) + Aq**2 * (s0 ^ s3) + Aq * s1 + s2
        Q[d,s] = s3 ^ s0 ^ d

  elif depth == 6:
    S = np.zeros([Aq,Aq**6],int)
    Q = np.zeros([Aq,Aq**6], int)
    for s in range(Aq**6):
      for d in range(Aq):
        ss = [(s / (Aq ** i)) % Aq for i in range(depth)]
        S[d,s] = Aq ** 5 * (ss[5] ^ d) + Aq**4 * (ss[0] ^ ss[5]) + Aq**3 * (ss[1]) + \
                 Aq ** 2 * (ss[2] ) +  Aq * ss[3] + ss[4]
        Q[d,s] = ss[5] ^ ss[0] ^ d
  else:
    S = np.zeros([Aq,Aq**depth],int)
    Q = np.zeros([Aq,Aq**depth], int)
    for s in range(Aq**depth):
      for d in range(Aq):
        ss = [(s / (Aq ** i)) % Aq for i in range(depth)]
        S[d,s] = Aq**(depth - 1) * (ss[depth - 1] ^ d) + \
        np.sum(np.asarray([Aq**(depth - i - 2) * ss[i] for i in range(depth - 1)]))
        Q[d,s] = ss[depth - 1] ^ ss[0] ^ d
  return FSM(S, Q)

def stream_encode(b, code, n, inter): 
  ''' Turbo code at given rate expressed by length(b)/n. The rate can range from 1 to 1/3. A
  parralely concatennated turbo code beginning with the systematic part is considered.

  Parameters:
  -----------
  b : array of ints
      Data vector of length k to be encoded. 
  code : Object
      Object describing the FSM used for the channel encoding. 
  n : int
      Length of the codeword.
  inter : array of ints 
      Interleaver. 

  Returns: 
  --------
  c : array of ints
      Turbo encoded data.

  See Also:
  ---------
  stream_decode

  Examples:
  ---------
  n = 7680
  k = 7680/2
  SNR = -2
  inter = np.arange(k)
  deinter = np.zeros(k, int)
  for i in range(0, k):
    deinter[inter[i]] = i

  code = create_code()
  b = np.random.randint(2, size=k)
  c = stream_encode(b, code, n, inter)
  s = np.array([-1,1])[c]
  x, sigma2w = AWGN(s, SNR)
  mu = demod(x, np.array([-1,1]), sigma2w, h = 1, norming = 0)
  mu_o = stream_decode(code, inter, deinter, mu, n, K)
  '''
  b_int = b[inter] 
  k = len(b)
  p = n - k
  map_inds = np.asarray([2 * i * k / p for i in range(p/2)])
  c1 = code.encode(b, 0)
  c2 = code.encode(b_int, 0)
  cd = np.vstack([c1[map_inds], c2[map_inds]]).flatten(1)
  return np.hstack([b, cd])

def stream_decode(code, inter, deinter, mu_i, n, K, Aq = 2): 
  '''
  Decode a stream encoded by the stream_encode function

  Parameters:
  -----------
  code : Object
      Object describing the FSM used for the channel encoding. 
  inter : array of ints 
      Interleaver. 
  deinter : array of ints 
      Deinterleaver. 
  mu_i : 2D array of floats 
      n x 2 array of input metric, where the first k symbols refers to systematic part.
  n : int
      Length of the codeword.
  K : int
      Number of iterations. 
  Aq : int
      Cardinality of the data symbol (implicitly assumed binary data)

  Returns: 
  --------
  mu_o : 2D array of floats
      n x 2 array of input metric.

  See Also:
  ---------
  stream_encode

  Examples:
  ---------
  # binary decoding
  n = 20
  k = 10
  SNR = 0
#  inter = np.arange(k)
  inter = np.random.permutation(k)
  deinter = np.zeros(k, int)
  for i in range(0, k):
    deinter[inter[i]] = i

  code = create_code()
  b = np.random.randint(2, size=k)
  c = stream_encode(b, code, n, inter)
  s = np.array([-1,1])[c]
  x, sigma2w = AWGN(s, SNR)
  mu = demod(x, np.array([-1,1]), sigma2w, h = 1, norming = 0)
  mu_o = stream_decode(code, inter, deinter, mu, n, K)


  # non-binary decoding
  q = 4 # Fourary alphabet
  n = 20
  k = 10 # code rate 1/2
  SNR = 2
  inter = np.arange(k)
  deinter = np.zeros(k, int)
  for i in range(0, k):
    deinter[inter[i]] = i

  code = create_code(4)
  b = np.random.randint(4, size=k)
  c = stream_encode(b, code, n, inter)
  s = np.array([-1,1j,1,-1j])[c] # QPSK mapper
  x, sigma2w = AWGN(s, SNR)
  mu = demod(x, np.array([-1,1j,1,-1j]), sigma2w, h = 1, norming = 0)
  mu_o = stream_decode(code, inter, deinter, mu, n, K, q)
  '''
  k = len(inter)
  p = n - k 
  mbi = mu_i[:k,:]
  mci = mu_i[k:,:]
  map_inds = np.asarray([2 * i * k / p for i in range(p)])
  map_inds2 = np.asarray([2 * i * k / p for i in range(p/2)])

  mci1t = np.ones([k, Aq], float)/Aq;  mci2t = np.ones([k, Aq], float)/Aq;
  mci1t[map_inds2, :] = mci[::2]
  mci2t[map_inds2, :] = mci[1::2]


  mci1 = mci1t.flatten()
  mci2 = mci2t.flatten()
  in1 = mbi.flatten()
  for i in range(0, K):
    out1 =  code.update(mci1, in1)
    oo = out1.reshape(k, Aq)
    in2 = (out1 * mbi.flatten()).reshape(k, Aq)[inter]
    in2 = (in2.flatten()/ (in2.sum(axis=1).repeat(Aq)))
    out2t = code.update(mci2, in2)
    out2 = out2t.reshape(k, Aq)[deinter]
    in1 = (out2 * mbi)
    in1 = (in1.flatten()/ (in1.sum(axis=1).repeat(Aq)))

  mco1, out1 = code.update_both(mci1, in1)
  oo = out1.reshape(k, Aq)
#    in2 = (out1 * mci1).reshape(N, 4)[inter]
  in2 = (out1 * mbi.flatten()).reshape(k, Aq)[inter]
  in2 = (in2.flatten()/ (in2.sum(axis=1).repeat(Aq)))
  mco2, out2t = code.update_both(mci2, in2)
  out2 = out2t.reshape(k, Aq)[deinter].flatten()

  out = (out1 * out2 * mbi.flatten()).reshape(k, Aq)
#  print out1.reshape(k, 2),  out2.reshape(k, 2)
  mbo = (out.flatten()/ (out.sum(axis=1).repeat(Aq))).reshape(k, Aq)

  mco = np.zeros([p, Aq], float)
  mco[::2, :] = mco1.reshape(k, Aq)[map_inds2, :]
  mco[1::2, :] = mco2.reshape(k, Aq)[map_inds2, :]

  return np.vstack([mbo, mco])

def test_streams(SNR, Aq=2):
  '''
  Method for testing turbo coding and decoding at given rate
  '''
  n = 76800
  k = 38900
  K = 45
#  inter = np.arange(k)
  inter = np.random.permutation(k)
  deinter = np.zeros(k, int)
  for i in range(0, k):
    deinter[inter[i]] = i

  const = np.exp(1j*2*np.pi*np.arange(Aq)/Aq) #Aq-PSK
  code = create_code(Aq)
  b = np.random.randint(Aq, size=k)
  c = stream_encode(b, code, n, inter)
  s = const[c]
  x, sigma2w = AWGN(s, SNR)
  mu = demod(x, const, sigma2w, h = 1, norming = 0)
  mu_o = stream_decode(code, inter, deinter, mu, n, K, Aq)
#  print mu_o, c, '\n', np.sum(np.argmax(mu_o, axis = 1) != c)
  print 'nerr:', np.sum(np.argmax(mu_o[:k,:], axis = 1) != b )
#  print inter, np.asarray([2 * i * k / (n-k) for i in range(n-k)]), np.asarray([2 * i * k / (n-k)  for i in range((n-k)/2)])

def P2P_higher_mod(Nbits, N, SNR, K = 10, K2 = 5, M = 1, r = 0.5, const = []):
  '''
  Run the turbo encoded P2P system in AWGN channel with binary data and a higher order
  constellation.

  Parameters:
  -----------
  Nbits : int
      Number of bits per one signal space symbol (Nbits (QAM) constellation).
  N : int
      Length of the frame in terms of signal space symbols (the binary codeword length is N * Nbits).
  K : int
      Number of iterations within the turbo decoder. 
  K2 : int
      Number of iterations between the turbo decoder and soft output demodulator. 
  r : float
      Rate of the code

  Returns: 
  --------
  nerr : int
      Number of errors. 

  See Also:
  ---------
  stream_encode, stream decode

  Examples:
  ---------
  P2P_higher_mod(4, 6000, 15, 10, 5, M = 1, r=0.75)
  '''

  n = N * Nbits # length of all data
  k = int(n * r)
  inter = np.random.permutation(k)
  deinter = np.zeros(k, int)
  for i in range(0, k):
    deinter[inter[i]] = i
  
  if const == []:
    const = QAM(Nbits)[0] 

  code = create_code()

  b = np.random.randint(2, size=(k))
  c = stream_encode(b, code, n , inter).reshape(N, Nbits, order='F')
  c_int = const_map(c)
  s = const[c_int]
  x, sigma2w = AWGN(s, SNR)
  muR = demod(x, const, sigma2w, h = 1, norming = 1)
  
  print 'uncoded nerr=', np.sum(c_int != np.argmax(muR, axis=1))                         # Verifica
  mu_o = np.ones([N, 2, Nbits]) * 0.5                       # A priory information
  for i in range(K2):
    mu = soft_demap(muR, mu_o, N, M, Nbits)                       # Update of soft message with a uniform priory info
 #   print mu, c
    mui = np.vstack([mu[:,0,:].flatten(1), mu[:,1,:].flatten(1)]).transpose()
#    est_b = np.argmax(mui[:k,:],axis=1)                              # Estimation of bits
 #   print i, 'th iteration, nerr=', np.sum(b != est_b)                         # Verifica
    dec = stream_decode(code, inter, deinter, mui, n, K)
    mu_o[:, 0, :] = dec[:,0].reshape(N, Nbits, order='F')
    mu_o[:, 1, :] = dec[:,1].reshape(N, Nbits, order='F')


  est_b = np.argmax(dec[:k,:],axis=1)                              # Estimation of bits
  print 'nerr=', np.sum(b != est_b)                         # Verifica
#  print b, est_b
#  print mu_o


def run_Encoded_MAC(Nb, Ns, N, SNR, K = 10, K2 = 5, M = 1, rb = 0.8, rs = 0.66):
  '''
  Runnig hierarchical MAC with channel encoding on GF2 -- one encoder per hierarchical stream
  '''

  code = create_code()
  ks = int(N * rs) * Ns
  kb = int(N * rb) * Nb
  ns = N * Ns
  nb = N * Nb
  bAs = np.random.randint(2, size=ks)
  bAb = np.random.randint(2, size=kb)

  bBs = np.random.randint(2, size=ks)
  bBb = np.random.randint(2, size=kb)

  inter_b = np.random.permutation(kb)
  deinter_b = np.zeros(kb, int)
  for i in range(0, kb):
    deinter_b[inter_b[i]] = i

  inter_s = np.random.permutation(ks)
  deinter_s = np.zeros(ks, int)
  for i in range(0, ks):
    deinter_s[inter_s[i]] = i
  
  if Ns > 0:
    cAs = stream_encode(bAs, code, ns, inter_s).reshape(N, Ns, order='F')
    cBs = stream_encode(bBs, code, ns, inter_s).reshape(N, Ns, order='F')
    cAs_int = const_map(cAs)
    cBs_int = const_map(cBs)
  if Nb > 0:
    cAb = stream_encode(bAb, code, nb, inter_b).reshape(N, Nb, order='F')
    cBb = stream_encode(bBb, code, nb, inter_b).reshape(N, Nb, order='F')
    cAb_int = const_map(cAb)
    cBb_int = const_map(cBb)

  if Ns == 0:
    muH = HDF_MAC_Signal_Space(Nb, Ns, SNR, 0, cAb_int, 0, cBb_int)
  elif Nb == 0:
    muH = HDF_MAC_Signal_Space(Nb, Ns, SNR, cAs_int, 0, cBs_int, 0)
  else:
    muH = HDF_MAC_Signal_Space(Nb, Ns, SNR, cAs_int, cAb_int, cBs_int, cBb_int)
#    muH = HDF_MAC_Signal_Space(Nb, Ns, SNR, cAs_int, cAb_int, cBs_int, cBb_int,1,0.3+1j)

  mu_o = np.ones([M * N, 2, Nb + 2*Ns]) * 0.5                       # A priory information
  for i in range(K2):
    mu = soft_demap(muH, mu_o, N, M, Nb + 2*Ns, norming = 1)
    
    if Ns > 0:
  # Superposed A branch
      mui = np.vstack([mu[:,0,:Ns].flatten(1), mu[:,1,:Ns].flatten(1)]).transpose()
      decAs = stream_decode(code, inter_s, deinter_s, mui, ns, K)
      mu_o[:, 0, :Ns] = decAs[:,0].reshape(N, Ns, order='F')
      mu_o[:, 1, :Ns] = decAs[:,1].reshape(N, Ns, order='F')

  # Superposed B branch
      mui = np.vstack([mu[:,0,Ns:(2*Ns)].flatten(1), mu[:,1,Ns:(2*Ns)].flatten(1)]).transpose()
      decBs = stream_decode(code, inter_s, deinter_s, mui, ns, K)
      mu_o[:, 0, Ns:(2*Ns)] = decBs[:,0].reshape(M * N, Ns, order='F')
      mu_o[:, 1, Ns:(2*Ns)] = decBs[:,1].reshape(M * N, Ns, order='F')
    
    if Nb > 0:
  # Basic B branch
      mui = np.vstack([mu[:,0,(2*Ns):].flatten(1), mu[:,1,(2*Ns):].flatten(1)]).transpose()
      decb = stream_decode(code, inter_b, deinter_b, mui, nb, K)
      mu_o[:, 0, (2*Ns):] = decb[:,0].reshape(M * N, Nb, order='F')
      mu_o[:, 1, (2*Ns):] = decb[:,1].reshape(M * N, Nb, order='F')

  if Ns > 0:
    est_bAs = np.argmax(decAs[:ks,:], axis=1)
    est_bBs = np.argmax(decBs[:ks,:], axis=1)
    print 'nerr As:', np.sum(est_bAs != bAs) 
    print 'nerr Bs:', np.sum(est_bBs != bBs)
  else: 
    est_bAs = []
    est_bBs = []
  if Nb > 0:
    est_bb = np.argmax(decb[:kb,:], axis=1)
    print 'nerr b:',  np.sum(est_bb != (bAb ^ bBb))
  else: 
    est_bb = []
  
  return est_bAs, est_bBs, est_bb
  
#  print est_bAs, bAs
#  print est_bBs, bBs
#  print est_bb, bAb ^ bBb
#  print mu_o

def run_Encoded_BC_GFq(Nb, Ns, N, SNR, K = 10, M = 1, r = 0.5):
  '''
  Run the BC channel on higher order field
  '''
  k = int(N * r)
  inter = np.random.permutation(k)
  deinter = np.zeros(k, int)
  for i in range(0, k):
    deinter[inter[i]] = i
  
  Nbits = 2*Ns + Nb
  const = QAM(Nbits)[0] 
  Aq = 2**Nbits
  code = create_code(Aq)
  b = np.random.randint(Aq, size=k)
  c = stream_encode(b, code, N, inter)
  s = const[c]
  x, sigma2w = AWGN(s, SNR)
  mu = demod(x, const, sigma2w, h = 1, norming = 0)
  mu_o = stream_decode(code, inter, deinter, mu, N, K, Aq)
#  print mu_o, c, '\n', np.sum(np.argmax(mu_o, axis = 1) != c)
  print 'nerr:', np.sum(np.argmax(mu_o[:k,:], axis = 1) != b )
#  print inter, np.asarray([2 * i * k / (n-k) for i in range(n-k)]), np.asarray([2 * i * k / (n-k)  for i in range((n-k)/2)])
  

def run_Encoded_BC_GF2(Nb, Ns, N, SNR, K = 10, K2 = 5, M = 1, r = 0.5):
  '''
  Run the BC channel on GF2 order field
  '''
  Nbits = 2*Ns + Nb
  const = QAM(Nbits)[0] 
  code = create_code()

  n = N * Nbits # length of all data
  k = int(n * r)
  inter = np.random.permutation(k)
  deinter = np.zeros(k, int)
  for i in range(0, k):
    deinter[inter[i]] = i
  
  b = np.random.randint(2, size=(k))
  c = stream_encode(b, code, n , inter).reshape(N, Nbits, order='F')
  c_int = const_map(c)
  s = const[c_int]
  x, sigma2w = AWGN(s, SNR)
  muR = demod(x, const, sigma2w, h = 1, norming = 1)
  
  print 'uncoded nerr=', np.sum(c_int != np.argmax(muR, axis=1))                         # Verifica
  mu_o = np.ones([N, 2, Nbits]) * 0.5                       # A priory information
  for i in range(K2):
    mu = soft_demap(muR, mu_o, N, M, Nbits)                       # Update of soft message with a uniform priory info
    mui = np.vstack([mu[:,0,:].flatten(1), mu[:,1,:].flatten(1)]).transpose()
    dec = stream_decode(code, inter, deinter, mui, n, K)
    mu_o[:, 0, :] = dec[:,0].reshape(N, Nbits, order='F')
    mu_o[:, 1, :] = dec[:,1].reshape(N, Nbits, order='F')

  est_b = np.argmax(dec[:k,:],axis=1)                              # Estimation of bits
  print 'nerr=', np.sum(b != est_b)                         # Verifica

def run_Encoded_HSI_GF2(Nb, Ns, N, SNR, K = 10, K2 = 5, M = 1, r = 0.5, alp = 0.0, hA = 1, hB = 1):
  '''
  Run the HSI channel on GF2 order field. Processing in destination B.
  '''
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h = 1)
  code = create_code(2,6)

  n = N * Nb # length of all data
  k = int(n * r)
  lenB = 2 ** Nb
  inter = np.random.permutation(k)
  deinter = np.zeros(k, int)
  for i in range(0, k):
    deinter[inter[i]] = i
  
  bA = np.random.randint(2, size=(k))
  cA = stream_encode(bA, code, n , inter).reshape(N, Nb, order='F')
  cA_int = const_map(cA)
  sA = basic_part[cA_int]

  bB = np.random.randint(2, size=(k))
  cB = stream_encode(bB, code, n , inter).reshape(N, Nb, order='F')
  cB_int = const_map(cB)
  sB = basic_part[cB_int]

  bAB = bA ^ bB
  cAB_int = cA_int ^ cB_int
  
  s = hA * sA + hB * alp * sB
  x, sigma2w = AWGN(s, SNR)
  
  muR = np.zeros([N, lenB], float)
  for j in range(lenB):
    act_range, = np.nonzero(j == (cA_int ^ cB_int))
    const = np.asarray([hA * basic_part[i] + alp * hB * basic_part[i ^ j]  for i in range(lenB)])
    for i in range(lenB):
#      muR[act_range, i] = np.exp(-np.abs(x[act_range] - const[i, act_range])**2/sigma2w )
      muR[act_range, i] = np.exp(-np.abs(x[act_range] - const[i])**2/sigma2w )

  if (np.min(muR) == 0):
    muR += 1e-100
  
  print 'uncoded nerr=', np.sum(cB_int  != ( cAB_int ^ np.argmax(muR, axis=1)))
  mu_o = np.ones([N, 2, Nb]) * 0.5 
  for i in range(K2):
    mu = soft_demap(muR, mu_o, N, M, Nb)     
    mui = np.vstack([mu[:,0,:].flatten(1), mu[:,1,:].flatten(1)]).transpose()
    dec = stream_decode(code, inter, deinter, mui, n, K)
    mu_o[:, 0, :] = dec[:,0].reshape(N, Nb, order='F')
    mu_o[:, 1, :] = dec[:,1].reshape(N, Nb, order='F')

  est_b = np.argmax(dec[:k,:],axis=1)   
  print 'nerr=', np.sum(bB != bAB ^ est_b)     
#  print bB, est_b


## Functions for system implementation

def run_Encoded_MAC_syst(Nb, Ns, N, SNR, bAs, bAb, bBs, bBb, code, inter_b, deinter_b, inter_s, \
    deinter_s, K = 10, K2 = 5, M = 1, rb = 0.8, rs = 0.66):
  '''
  Runnig hierarchical MAC with channel encoding on GF2 -- one encoder per hierarchical stream. This
  function is suited for overall system emulation
  '''

  ks = int(N * rs) * Ns
  kb = int(N * rb) * Nb
  ns = N * Ns
  nb = N * Nb
  
  if Ns > 0:
    cAs = stream_encode(bAs, code, ns, inter_s).reshape(N, Ns, order='F')
    cBs = stream_encode(bBs, code, ns, inter_s).reshape(N, Ns, order='F')
    cAs_int = const_map(cAs)
    cBs_int = const_map(cBs)
  if Nb > 0:
    cAb = stream_encode(bAb, code, nb, inter_b).reshape(N, Nb, order='F')
    cBb = stream_encode(bBb, code, nb, inter_b).reshape(N, Nb, order='F')
    cAb_int = const_map(cAb)
    cBb_int = const_map(cBb)

  if Ns == 0:
    muH = HDF_MAC_Signal_Space(Nb, Ns, SNR, 0, cAb_int, 0, cBb_int)
  elif Nb == 0:
    muH = HDF_MAC_Signal_Space(Nb, Ns, SNR, cAs_int, 0, cBs_int, 0)
  else:
    muH = HDF_MAC_Signal_Space(Nb, Ns, SNR, cAs_int, cAb_int, cBs_int, cBb_int)
#    muH = HDF_MAC_Signal_Space(Nb, Ns, SNR, cAs_int, cAb_int, cBs_int, cBb_int,1, 1j)

  mu_o = np.ones([M * N, 2, Nb + 2*Ns]) * 0.5                       # A priory information
  for i in range(K2):
    mu = soft_demap(muH, mu_o, N, M, Nb + 2*Ns, norming = 1)
    
    if Ns > 0:
  # Superposed A branch
      mui = np.vstack([mu[:,0,:Ns].flatten(1), mu[:,1,:Ns].flatten(1)]).transpose()
      decAs = stream_decode(code, inter_s, deinter_s, mui, ns, K)
      mu_o[:, 0, :Ns] = decAs[:,0].reshape(N, Ns, order='F')
      mu_o[:, 1, :Ns] = decAs[:,1].reshape(N, Ns, order='F')

  # Superposed B branch
      mui = np.vstack([mu[:,0,Ns:(2*Ns)].flatten(1), mu[:,1,Ns:(2*Ns)].flatten(1)]).transpose()
      decBs = stream_decode(code, inter_s, deinter_s, mui, ns, K)
      mu_o[:, 0, Ns:(2*Ns)] = decBs[:,0].reshape(M * N, Ns, order='F')
      mu_o[:, 1, Ns:(2*Ns)] = decBs[:,1].reshape(M * N, Ns, order='F')
    
    if Nb > 0:
  # Basic B branch
      mui = np.vstack([mu[:,0,(2*Ns):].flatten(1), mu[:,1,(2*Ns):].flatten(1)]).transpose()
      decb = stream_decode(code, inter_b, deinter_b, mui, nb, K)
      mu_o[:, 0, (2*Ns):] = decb[:,0].reshape(M * N, Nb, order='F')
      mu_o[:, 1, (2*Ns):] = decb[:,1].reshape(M * N, Nb, order='F')

  if Ns > 0:
    est_bAs = np.argmax(decAs[:ks,:], axis=1)
    est_bBs = np.argmax(decBs[:ks,:], axis=1)
    print 'MAC nerr As:', np.sum(est_bAs != bAs) 
    print 'MAC nerr Bs:', np.sum(est_bBs != bBs)
  else: 
    est_bAs = []
    est_bBs = []
  if Nb > 0:
    est_bb = np.argmax(decb[:kb,:], axis=1)
    print 'MAC nerr b:',  np.sum(est_bb != (bAb ^ bBb))
  else: 
    est_bb = []
  
  return est_bAs, est_bBs, est_bb
  
#  print est_bAs, bAs
#  print est_bBs, bBs
#  print est_bb, bAb ^ bBb
#  print mu_o

def run_Encoded_BC_GF2_syst(Nb, Ns, N, SNR, b, r, code, inter, deinter, const, K = 10, K2 = 5, M = 1):
  '''
  Run the BC channel on GF2 order field
  '''
  Nbits = 2*Ns + Nb
  n = Nbits * N
  k = int(n * r)
  
  c = stream_encode(b, code, n , inter).reshape(N, Nbits, order='F')
  c_int = const_map(c)
  s = const[c_int]
  x, sigma2w = AWGN(s, SNR)
  muR = demod(x, const, sigma2w, h = 1, norming = 1)
  
#  print 'uncoded nerr=', np.sum(c_int != np.argmax(muR, axis=1))                         # Verifica
  mu_o = np.ones([N, 2, Nbits]) * 0.5                       # A priory information
  for i in range(K2):
    mu = soft_demap(muR, mu_o, N, M, Nbits)                       # Update of soft message with a uniform priory info
    mui = np.vstack([mu[:,0,:].flatten(1), mu[:,1,:].flatten(1)]).transpose()
    dec = stream_decode(code, inter, deinter, mui, n, K)
    mu_o[:, 0, :] = dec[:,0].reshape(N, Nbits, order='F')
    mu_o[:, 1, :] = dec[:,1].reshape(N, Nbits, order='F')

  est_b = np.argmax(dec[:k,:],axis=1)                              # Estimation of bits
  print 'BC nerr:', np.sum(b != est_b)                         # Verifica
  return est_b


def run_Encoded_HSI_GF2_syst(Nb, Ns, N, SNR, bAs, bAb, bBs, bBb, code, inter_b, deinter_b, inter_s, \
    deinter_s, K, K2, rb, rs, hA, hB, alp, est_bR):
  '''
  Run the HSI channel on GF2 order field. Processing in destination B.
  '''
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h = 1)
  super_part = const_superposed(Nb, Ns, 'XOR')
  ks = int(N * rs) * Ns
  kb = int(N * rb) * Nb
  ns = N * Ns
  nb = N * Nb
  lenB = 2 ** Nb
  
  if Ns > 0:
    cAs = stream_encode(bAs, code, ns, inter_s).reshape(N, Ns, order='F')
    cBs = stream_encode(bBs, code, ns, inter_s).reshape(N, Ns, order='F')
    cAs_int = const_map(cAs)
    cBs_int = const_map(cBs)
  if Nb > 0:
    cAb = stream_encode(bAb, code, nb, inter_b).reshape(N, Nb, order='F')
    cBb = stream_encode(bBb, code, nb, inter_b).reshape(N, Nb, order='F')
    cAb_int = const_map(cAb)
    cBb_int = const_map(cBb)
  
  if Ns == 0:
    cAs_int = 0
    cBs_int = 0
  cA_int = (cAs_int << Nb) + cAb_int
  cB_int = (cBs_int << Nb) + cBb_int

  sA = sourceA_const[cA_int]
  sB = sourceB_const[cB_int]
  
# Destination B observation in the firts time slot
  (x,sigma2w) = AWGN(hA*sA + alp*hB*sB, SNR)

  est_bAs = est_bR[:ks] 
  est_bBs = est_bR[ks:(2*ks)] 
  est_bAB = est_bR[(2*ks):] 
  c_AB = stream_encode(est_bAB, code, nb, inter_b).reshape(N, Nb, order='F')
  c_AB_int = const_map(c_AB)

  if Ns > 0:
    c_As = stream_encode(est_bAs, code, ns, inter_s).reshape(N, Ns, order='F')
    c_Bs = stream_encode(est_bBs, code, ns, inter_s).reshape(N, Ns, order='F')
    c_As_int = const_map(c_As)
    c_Bs_int = const_map(c_Bs)

    sAs = super_part[c_As_int]
    sBs = 1j*super_part[c_Bs_int]
    # Interference cancellation
    y = x - hA * sAs - alp*hB * sBs
  
  else:
    y = x
#  plt.plot(np.real(basic_part), np.imag(basic_part), 'gp', ms = 20)
#  plt.plot(np.real(x), np.imag(x), 'xr', ms = 10)
#  plt.plot(np.real(y), np.imag(y), 'ko', ms = 5)
#  plt.show()
  muR = np.zeros([N, lenB], float)
  for j in range(lenB):
    act_range, = np.nonzero(j == c_AB_int)
    const = np.asarray([hA * basic_part[i] + alp * hB * basic_part[i ^ j]  for i in range(lenB)])
    for i in range(lenB):
#      muR[act_range, i] = np.exp(-np.abs(x[act_range] - const[i, act_range])**2/sigma2w )
      muR[act_range, i] = np.exp(-np.abs(x[act_range] - const[i])**2/sigma2w )

  if (np.min(muR) == 0):
    muR += 1e-100
  
#  print 'uncoded nerr=', np.sum(cBb_int  != ( c_AB_int ^ np.argmax(muR, axis=1)))
  mu_o = np.ones([N, 2, Nb]) * 0.5 
  for i in range(K2):
    mu = soft_demap(muR, mu_o, N, M, Nb)     
    mui = np.vstack([mu[:,0,:].flatten(1), mu[:,1,:].flatten(1)]).transpose()
    dec = stream_decode(code, inter_b, deinter_b, mui, nb, K)
    mu_o[:, 0, :] = dec[:,0].reshape(N, Nb, order='F')
    mu_o[:, 1, :] = dec[:,1].reshape(N, Nb, order='F')

  est_b = np.argmax(dec[:kb,:],axis=1)   
  print 'HSI nerr=', np.sum(bBb != est_bAB ^ est_b)     
  return est_b
#  print bB, est_b


def run_overall_system(N, Nb, Ns, rb, rs, gMAC = 20, gBC = 20, gHSI = 20,  K=15, K2 = 5):
  code = create_code()
  ks = int(N * rs) * Ns
  kb = int(N * rb) * Nb
  ns = N * Ns
  nb = N * Nb
  bAs = np.random.randint(2, size=ks)
  bAb = np.random.randint(2, size=kb)

  bBs = np.random.randint(2, size=ks)
  bBb = np.random.randint(2, size=kb)

  inter_b = np.random.permutation(kb)
  deinter_b = np.zeros(kb, int)   
  for i in range(0, kb):
    deinter_b[inter_b[i]] = i

  inter_s = np.random.permutation(ks)
  deinter_s = np.zeros(ks, int)
  for i in range(0, ks):
    deinter_s[inter_s[i]] = i

  nBC = N * (Nb + 2*Ns)
  kBC =  2*ks + kb
  rBC = float(kBC) / nBC
  inter_BC = np.random.permutation(kBC)
  deinter_BC = np.zeros(kBC, int)
  for i in range(0, kBC):
    deinter_BC[inter_BC[i]] = i
  
  constBC = QAM(Nb+2*Ns)[0]
 
  # Run MAC
  est_bAs, est_bBs, est_bb = run_Encoded_MAC_syst(Nb, Ns, N, gMAC, bAs, bAb, bBs, bBb, code,\
                             inter_b, deinter_b, inter_s, deinter_s, K, K2, 1, rb, rs)
  bR = np.hstack([ est_bAs, est_bBs, est_bb])
  # Run BC
  est_bR = run_Encoded_BC_GF2_syst(Nb, Ns, N, gBC, bR, rBC, code, inter_BC, deinter_BC, constBC, K = 10, K2 = 5, M = 1)

  # Run HSI
  if Nb > 0:
    est_bB = run_Encoded_HSI_GF2_syst(Nb, Ns, N, gHSI, bAs, bAb, bBs, bBb, code, inter_b, deinter_b, inter_s, \
      deinter_s, K, K2,rb , rs , 1 , 1 , 0, est_bR)


def P2P_test(N, SNR, K = 1):
#  inter = np.random.permutation(2*N)
  inter = np.arange(2*N)
  deinter = np.zeros(2*N, int)
  for i in range(0, 2*N):
    deinter[inter[i]] = i
  
  code = create_code()
  b = np.random.randint(2,size=N)
  c = code.encode(b,0)
  
  vec_stream = np.vstack([b[:N/2],b[N/2:],c[:N/2],c[N/2:]]).transpose()

  stream_int = interleave2D(inter, vec_stream)

  c_int = const_map(stream_int)

  const = QAM(4)[0] 

  s = const[c_int]
  (x,sigma2w) = AWGN(s, SNR)
  
  muR = demod(x, const, sigma2w)

  print 'Uncoded nerr:',np.sum(np.argmax(muR, axis=1) != c_int)
  
# SoDeM
  mu_o = np.ones([N/2, 2, 4]) * 0.25                       # A priory information
  for k in range(0, K):
    print k
    mu = soft_demap(muR, mu_o, N/2, 1, 4)                 # Update of soft message with a uniform priory info
    vec = interleave2D_metric(deinter, mu)
  
#    print vec, '\n\n', vec_stream
    apr = np.vstack([vec[:,:,0], vec[:,:,1]]).flatten()
    obs = np.vstack([vec[:,:,2], vec[:,:,3]]).flatten()
    o_u, a_u = code.update_both(obs, apr)
    vec[:,:,0] = a_u.reshape(N,2)[:N/2,:]
    vec[:,:,1] = a_u.reshape(N,2)[N/2:,:]
    vec[:,:,2] = o_u.reshape(N,2)[:N/2,:]
    vec[:,:,3] = o_u.reshape(N,2)[N/2:,:]
    mu_o = interleave2D_metric(inter, vec)
#    print a_u.reshape(N,2),'\n\n', o_u.reshape(N,2), '\n\n', b, c

#    vec2 = np.vstack([a_u.reshape(N,2), o_u.reshape(N,2)]) 
#    print vec2
#    mu_o = np.vstack([vec2[:,0][deinter],  vec2[:,1][deinter]]).transpose()
 
    nerr = np.sum(np.argmax(a_u.reshape(N,2),  axis=1) != b)
    print 'Coded nerr:', nerr

def GF2_run(n, k, SNR_vals, M, K, K2 = 1, interleaving = 0): # HSI link
  H = LDPC_lib.LDPC(n,k)
  H.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-k)))
  H.prepare_decoder_C()
#  QPSK = np.exp(1j * np.arange(0,4)*np.pi/2)
  QPSK = np.array([-0.66666667-0.66666667j, -0.66666667+0.66666667j, 0.66666667-0.66666667j,  0.66666667+0.66666667j])
  
  inter = np.random.permutation(M*n)
  deinter = np.zeros(M*n, int)
  for i in range(0, M*n):
    deinter[inter[i]] = i

  b = np.random.randint(2,size=[M,k])
  c = H.encode(b).flatten()
  if interleaving:
    c_int = c[inter]
    c_map = c_int[0::2]*2+c_int[1::2]
  else:
    c_map = c[0::2]*2+c[1::2]
  
#  c_map = c[:,0::2]*2+c[:,1::2];
  s = QPSK[c_map]
#  print inter, deinter, c

  for SNR in SNR_vals:
    (x,sigma2w) = AWGN(s, SNR)
    x = x.flatten()
    muR = np.zeros([n * M / 2, 4], float) # 4ARY constellation fixed

    for i in range(0,n * M / 2):
      muR[i,:] = np.asarray([np.exp(-np.abs(x[i] - s0)**2/sigma2w) for s0 in QPSK]).flatten()

    if (np.min(muR) == 0):
      muR += 1e-100

#    est_c = gen_decode_mod(muR, H, M, 2, K, K2).flatten()
#    est_cmap = np.asarray([est_c[i::N]*(1 << (N - i - 1)) for i in range(N)]).sum(axis = 0)
#    print np.sum(est_cmap != c_map)
    print GF2_decode(muR, H, M, K, c, inter, deinter, K2, interleaving)


def gen_decode(muR, H, M, N, K, K2 = 1): # Suppose N binary streams 
  mu_o = np.ones([M * H[0].n, 2, N]) * 0.5 
#  print muR
  for k in range(0, K2):    
    mu = np.zeros([H[0].n * M, 2, N], float)
# SODEM update using a prior info
    for i in range(1 << N):
      inds = dec2bin(i, N)
      for j in range(N):
        temp = np.asarray([mu_o[:, inds[l], l] for l in range(N) if l != j])
#        print temp
#        print temp.prod(axis = 0)
        mu[:, inds[j], j] += muR[:, i] * temp.prod(axis = 0)
# Decode all branches        
    for i in range(N):
      LLR_vec = np.log(mu[:,1,i]/mu[:,0,i])
#      print LLR_vec
      dec = np.zeros(LLR_vec.shape,dtype=float)
      soft_out = np.zeros(LLR_vec.shape,dtype=float)
      t1 = np.zeros([M ,H[i].num_edges],dtype=float).flatten()
      t2 = np.zeros([M ,H[i].num_edges],dtype=float).flatten()
      _LDPC_dec_lib.run_FG_SPA_ef_soft_out_func(t1, t2, dec, -LLR_vec, soft_out, \
        H[i].FN_deg_distr, H[i].FN_map, H[i].FN_map_for_C, H[i].VN_deg_distr, H[i].VN_map_for_C, M , K, 1)
      soft_out[np.nonzero(np.isnan(soft_out))] = 0 # avoiding nans
#      print soft_out
      mu_o[:,1,i] =  H[i].LLR2pr1(soft_out)
      mu_o[:,0,i] =  1 - mu_o[:,1,i]
#    print mu_o[:,0,:]

#  gen_decode_mod(muR, H, M, N, K, K2 = 1)
  return np.argmax(mu_o, axis=1)

def test_gen_dec(SNR, M = 1, N = 2, k = 32400, K = 45, K2 = 15):
  n =64800
#  k = 32400
#  k = 48600
#  n = 6
#  k = 3
  H = LDPC_lib.LDPC(n, k)
  H.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-k)))
  H.prepare_decoder_C()

  const = QAM(N)[0]
#  const = np.array([-0.66666667-0.66666667j, -0.66666667+0.66666667j, 0.66666667-0.66666667j, 0.66666667+0.66666667j])

  b =  np.random.randint(2,size=[M * N, k])
  c = H.encode(b).flatten()
  c_map = np.asarray([c[i::N]*(1 << (N - i - 1)) for i in range(N)]).sum(axis = 0)
  s = const[c_map]

  (x,sigma2w) = AWGN(s, SNR)

  muR = np.zeros([n * M, 2**N], float)
  for i in range(0, 2 ** N):
    muR[:, i] = np.exp(-np.abs(x - const[i])**2/sigma2w)
  
  est_c = gen_decode_mod(muR, H, M, N, K, K2).flatten()
  est_cmap = np.asarray([est_c[i::N]*(1 << (N - i - 1)) for i in range(N)]).sum(axis = 0)
  nerr = np.sum(c_map != est_cmap)
  return float(nerr)/H.n/M


def gen_decode_mod(muR, H, M, N, K, K2 = 1): # Suppose N binary streams 
  """ Decoder that should properly decode n*Hn blocks with encoding process as

  b =  np.random.randint(2,size=[M * N, k])
  c  = H.encode(b).flatten()
  c_map = np.asarray([c[i::N]*(1 << (N - i - 1)) for i in range(N)]).sum(axis = 0)
  s = const[c_map]

  for i in range(0, 2 ** N):
    muR[:, i] = np.exp(-np.abs(x - const[i])**2/sigma2w)

  """

  mu_o = np.ones([M * H.n, 2, N]) * 0.5 
#  print muR
  for k in range(0, K2):    
    mu = np.zeros([H.n * M, 2, N], float)
# SODEM update using a prior info
    for i in range(1 << N):
      inds = dec2bin(i, N)
      for j in range(N):
        temp = np.asarray([mu_o[:, inds[l], l] for l in range(N) if l != j])
#        print temp
#        print temp.prod(axis = 0)
        mu[:, inds[j], j] += muR[:, i] * temp.prod(axis = 0)
# Decode all branches        
    LLR = np.log(mu[:,1,:]/mu[:,0,:])
    tt =  LLR.flatten().reshape(N, M * H.n)
    for i in range(N):
      LLR_vec = tt[i,:]
#np.log(mu[:,1,i]/mu[:,0,i])
#      print LLR_vec
      dec = np.zeros(LLR_vec.shape,dtype=float)
      soft_out = np.zeros(LLR_vec.shape,dtype=float)
      t1 = np.zeros([M ,H.num_edges],dtype=float).flatten()
      t2 = np.zeros([M ,H.num_edges],dtype=float).flatten()
      _LDPC_dec_lib.run_FG_SPA_ef_soft_out_func(t1, t2, dec, -LLR_vec, soft_out, \
        H.FN_deg_distr, H.FN_map, H.FN_map_for_C, H.VN_deg_distr, H.VN_map_for_C, M , K, 1)
      soft_out[np.nonzero(np.isnan(soft_out))] = 0 # avoiding nans
#      print soft_out
      mu_o[:,1,i] =  H.LLR2pr1(soft_out)
      mu_o[:,0,i] =  1 - mu_o[:,1,i]
#    print mu_o[:,0,:]
  est = np.argmax(mu_o, axis=1).transpose()
  return est


def BC_decode(muR, Hb, Hs, M, Nb, Ns, K, K2 = 1):
  musA_o = np.ones([M * Hs.n, 1 << Ns]) * 1 / float(1 << Ns)
  musB_o = np.ones([M * Hs.n, 1 << Ns]) * 1 / float(1 << Ns)
  mub_o = np.ones([M * Hb.n, 1 << Nb]) * 1 / float(1 << Nb)
  for k in range(0, K2):    
#    print np.min(np.abs(0.5-musA_o)), np.min(np.abs(0.5-musB_o)), np.min(np.abs(0.25-mub_o))
    musA = np.zeros([Hs.n * M, 1 << Ns], float)
    musB = np.zeros([Hs.n * M, 1 << Ns], float)
    mub = np.zeros([Hb.n * M, 1 << Nb], float)
# SODEM update using a prior info
    for i in range(1 << (Nb + 2 * Ns)):
      inds = dec2bin(i, (Nb + 2 * Ns))
      isA = bin2dec(inds[:Ns]) 
      isB = bin2dec(inds[Ns:(2*Ns)]) 
      ib  = bin2dec(inds[2*Ns:]) 
      musA[:, isA] += muR[:, i] * musB_o[:, isB] * mub_o[:, ib]
      musB[:, isB] += muR[:, i] * musA_o[:, isA] * mub_o[:, ib]
      mub[: , ib ] += muR[:, i] * musA_o[:, isA] * musB_o[:, isB]

## basic branch
    musb = np.zeros([Hb.n * M * Nb, 2])
    musb_o = np.zeros([Hb.n * M * Nb, 2])
    for i in range(1 << Nb):
      inds = dec2bin(i, Nb)
      for j in range(Nb):
        musb[j::Nb, inds[j]] += mub[:, i]
    LLR_vec = np.log(musb[:,1]/musb[:,0])
    dec = np.zeros(LLR_vec.shape,dtype=float)
    soft_out = np.zeros(LLR_vec.shape,dtype=float)
    t1 = np.zeros([M * Nb,Hb.num_edges],dtype=float).flatten()
    t2 = np.zeros([M * Nb,Hb.num_edges],dtype=float).flatten()
    _LDPC_dec_lib.run_FG_SPA_ef_soft_out_func(t1, t2, dec, -LLR_vec, soft_out, \
      Hb.FN_deg_distr, Hb.FN_map, Hb.FN_map_for_C, Hb.VN_deg_distr, Hb.VN_map_for_C, M * Nb, K, 1)
    soft_out[np.nonzero(np.isnan(soft_out))] = 0 # avoiding nans
    musb_o[:,1] =  Hb.LLR2pr1(soft_out)
    musb_o[:,0] =  1 - musb_o[:,1]
    mub_ot =  np.ones([M * Hb.n, 1 << Nb], float) 
    for i in range(1 << Nb):
      inds = dec2bin(i, Nb)
      for j in range(Nb):
        mub_ot[:, i] = mub_ot[:, i] * musb_o[j::Nb, inds[j]]
    mub_o = mub_ot/(np.array([mub_ot.sum(axis=1)]).repeat(1 << Nb,axis=0)).transpose() # norming


# superposed A branch
    musA2 = np.zeros([Hs.n * M * Ns, 2])
    musA2_o = np.zeros([Hs.n * M * Ns, 2])
    for i in range(1 << Ns):
      inds = dec2bin(i, Ns)
      for j in range(Ns):
        musA2[j::Ns, inds[j]] += musA[:, i]
    LLR_vec = np.log(musA2[:,1]/musA2[:,0])
    dec = np.zeros(LLR_vec.shape,dtype=float)
    soft_out = np.zeros(LLR_vec.shape,dtype=float)
    t1 = np.zeros([M * Ns,Hs.num_edges],dtype=float).flatten()
    t2 = np.zeros([M * Ns,Hs.num_edges],dtype=float).flatten()
    _LDPC_dec_lib.run_FG_SPA_ef_soft_out_func(t1, t2, dec, -LLR_vec, soft_out, \
      Hs.FN_deg_distr, Hs.FN_map, Hs.FN_map_for_C, Hs.VN_deg_distr, Hs.VN_map_for_C, M * Ns, K, 1)
    soft_out[np.nonzero(np.isnan(soft_out))] = 0 # avoiding nans
    musA2_o[:,1] =  Hs.LLR2pr1(soft_out)
    musA2_o[:,0] =  1 - musA2_o[:,1]
    musA_ot =  np.ones([M * Hs.n, 1 << Ns], float) 
    for i in range(1 << Ns):
      inds = dec2bin(i, Ns)
      for j in range(Ns):
        musA_ot[:, i] = musA_ot[:, i] * musA2_o[j::Ns, inds[j]]
    musA_o = musA_ot/(np.array([musA_ot.sum(axis=1)]).repeat(1 << Ns,axis=0)).transpose() # norming
  
# superposed B branch
    musB2 = np.zeros([Hs.n * M * Ns, 2])
    musB2_o = np.zeros([Hs.n * M * Ns, 2])
    for i in range(1 << Ns):
      inds = dec2bin(i, Ns)
      for j in range(Ns):
        musB2[j::Ns, inds[j]] += musB[:, i]
    LLR_vec = np.log(musB2[:,1]/musB2[:,0])
    dec = np.zeros(LLR_vec.shape,dtype=float)
    soft_out = np.zeros(LLR_vec.shape,dtype=float)
    t1 = np.zeros([M * Ns,Hs.num_edges],dtype=float).flatten()
    t2 = np.zeros([M * Ns,Hs.num_edges],dtype=float).flatten()
    _LDPC_dec_lib.run_FG_SPA_ef_soft_out_func(t1, t2, dec, -LLR_vec, soft_out, \
      Hs.FN_deg_distr, Hs.FN_map, Hs.FN_map_for_C, Hs.VN_deg_distr, Hs.VN_map_for_C, M * Ns, K, 1)
    soft_out[np.nonzero(np.isnan(soft_out))] = 0 # avoiding nans
    musB2_o[:,1] =  Hs.LLR2pr1(soft_out)
    musB2_o[:,0] =  1 - musB_o[:,1]
    musB_ot =  np.ones([M * Hs.n, 1 << Ns], float) 
    for i in range(1 << Ns):
      inds = dec2bin(i, Ns)
      for j in range(Ns):
        musB_ot[:, i] = musB_ot[:, i] * musB2_o[j::Ns, inds[j]]
    musB_o = musB_ot/(np.array([musB_ot.sum(axis=1)]).repeat(1 << Ns,axis=0)).transpose() # norming

  return np.argmax(musA_o, axis=1),  np.argmax(musB_o, axis=1), np.argmax(mub_o, axis=1) 

def run_BC(SNR_vals, M, K, K2 = 1): 
  n = 64800
  kb = 48600
  ks = 43200
#  kb = 32400
#  ks = 32400
#  n = 6
#  kb = 3
#  ks = 3
  Nb = 2
  Ns = 1
  const = QAM(2*Ns + Nb)[0][np.array([0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15])]

  Hs = LDPC_lib.LDPC(n, ks)
  Hs.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-ks)))
  Hs.prepare_decoder_C()
  bsA = np.random.randint(2,size=[M * Ns, ks])
  bsB = np.random.randint(2,size=[M * Ns, ks])
  csA = Hs.encode(bsA).flatten()
  csB = Hs.encode(bsB).flatten()

  Hb = LDPC_lib.LDPC(n, kb)
  Hb.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-kb)))
  Hb.prepare_decoder_C()  
#  bb =  np.random.randint(2,size=[M * Nb, kb])
#  cb  = Hb.encode(bb).flatten()
  bb1 =  np.random.randint(2,size=[M, kb])
  cb1  = Hb.encode(bb1).flatten()
  bb2 =  np.random.randint(2,size=[M, kb])
  cb2  = Hb.encode(bb2).flatten()

#  c_map = csA*8 + csB*4 + cb[0::2]*2 + cb[1::2] # fixed for Nb=2, Ns=1
  c_map = csA*8 + csB*4 + cb1*2 + cb2 # fixed for Nb=2, Ns=1

#  c_map = c[:,0::2]*2+c[:,1::2];
  s = const[c_map]
  h = np.random.rayleigh(np.sqrt(2)/2, size = M * n) * np.exp(1j * np.random.uniform(0,2*np.pi, size = M * n))

  for SNR in SNR_vals:
    (x,sigma2w) = AWGN(h*s, SNR)
    x = x.flatten()
    muR = np.zeros([n * M, 2 ** (Nb + 2*Ns)], float) # 4ARY constellation fixed

#    for i in const:
#        plt.plot(np.real(i), np.imag(i), 'bx', ms=10)
#    for i in x:
#        plt.plot(np.real(i), np.imag(i), 'or', ms=5)
#    plt.show()

    for i in range(0, 2 ** (Nb + 2*Ns)):
      muR[:, i] = np.exp(-np.abs(x - h*const[i])**2/sigma2w) 

    if (np.min(muR) == 0):
      muR += 1e-100
   
    nerr = (np.argmax(muR, axis=1) != c_map).sum()
    print nerr # uncoded 
#    est_csA, est_csB, est_cb  = BC_decode(muR, Hb, Hs, M, Nb, Ns, K, K2)
#    est_c = est_csA*8 + est_csB*4 + est_cb

    est  = gen_decode(muR, [Hs,Hs,Hb,Hb], M, Nb+2*Ns, K, K2)
    est_cc = est[:,0]*8 + est[:,1]*4 + est[:,2]*2 + est[:,3]
#    print est, est_cc, '\n', np.asarray([dec2bin(i,4) for i in est_c]), est_c,'\n', c_map

#    nerr_coded = (est_c != c_map).sum()
    nerr_coded1 = (est_cc != c_map).sum()
    print nerr_coded1 #, nerr_coded

def test_dec(n, k, M = 1, SNR = 0, K = 30): # Test properties of a given LDPC code
  H = LDPC_lib.LDPC(n,k)
  H.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-k)))
  H.prepare_decoder_C()
  b = np.random.randint(2, size=(M, k))
  c = H.encode(b)
  s = c*2-1 #BPSK
  (x,sigma2w) = AWGN(s, SNR)
  x = x.flatten()
  
  mu = np.zeros([ len(x), 2])
  for i in range(2):
    mu[:,i] = np.exp(-np.abs(x - 2*i + 1)**2/sigma2w)

  if (np.min(mu) == 0):
    mu += 1e-100
  
#  print mu
  LLR_vec = np.log(mu[:,1]/mu[:,0])
  dec = np.zeros(LLR_vec.shape,dtype=float)
  soft_out = np.zeros(LLR_vec.shape,dtype=float)
  t1 = np.zeros([M,H.num_edges],dtype=float).flatten()
  t2 = np.zeros([M,H.num_edges],dtype=float).flatten()
  _LDPC_dec_lib.run_FG_SPA_ef_soft_out_func(t1, t2, dec, -LLR_vec, soft_out, \
    H.FN_deg_distr, H.FN_map, H.FN_map_for_C, H.VN_deg_distr, H.VN_map_for_C, M, K, 1)
  soft_out[np.nonzero(np.isnan(soft_out))] = 0 # avoiding nans
  dec[np.nonzero(np.isnan(dec))] = 0
  est = (dec < 0)
  nerr =  np.sum(est != c.flatten())
  Pbe = float(nerr)/ (k * M)
#  print dec, est+1-1
  return nerr, Pbe #, np.sum(c_est != c.flatten())



def run_MAC(SNR_vals, M, K, K2 = 1): 
  n = 64800
  kb = 48600
  ks = 43200
#  n = 6
#  kb = 3
#  ks = 3
  Nb = 2
  Ns = 1
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h = 1)

  Hs = LDPC_lib.LDPC(n, ks)
  Hs.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-ks)))
  Hs.prepare_decoder_C()
  bsA = np.random.randint(2,size=[M * Ns, ks])
  bsB = np.random.randint(2,size=[M * Ns, ks])
  csA = Hs.encode(bsA).flatten()
  csB = Hs.encode(bsB).flatten()

  Hb = LDPC_lib.LDPC(n, kb)
  Hb.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-kb)))
  Hb.prepare_decoder_C()  
  bbA =  np.random.randint(2,size=[M * Nb, kb])
  bbB =  np.random.randint(2,size=[M * Nb, kb])
  cbA  = Hb.encode(bbA).flatten()
  cbB  = Hb.encode(bbB).flatten()

  c_mapA = csA*4 + cbA[0::2]*2 + cbA[1::2] # fixed for Nb = 2, Ns = 1
  sA = sourceA_const[c_mapA]

  c_mapB = csB*4 + cbB[0::2]*2 + cbB[1::2] # fixed for Nb = 2, Ns = 1
  sB = sourceB_const[c_mapB]
 
  c_map = csA*8 + csB*4 + (cbA[0::2]  ^ cbB[0::2])*2 + (cbA[1::2]  ^ cbB[1::2])

  for SNR in SNR_vals:
    (x,sigma2w) = AWGN(sA + sB, SNR)
    x = x.flatten()
    muR = np.zeros([n * M, 2 ** (2*Nb + 2*Ns)], float)
    muH = np.zeros([n * M, 2 ** (Nb + 2*Ns)], float) 

    for i in range(0, 1 << (2*Nb + 2*Ns)):
      muR[:, i] = np.exp(-np.abs(x - relay_const[i])**2/sigma2w) 
      inds = dec2bin(i,6)
      hier_ind = bin2dec(np.array([inds[0], inds[3], inds[1]^inds[4], inds[2]^inds[5]]))
      muH[:, hier_ind] += muR[:, i]    

    if (np.min(muH) == 0):
      muH += 1e-100


    nerr = (np.argmax(muH, axis=1) != c_map).sum()
    print nerr
    est_csA, est_csB, est_cb  = BC_decode(muH, Hb, Hs, M, Nb, Ns, K, K2)
    est_c = est_csA*8 + est_csB*4 + est_cb

    nerr_coded = (est_c != c_map).sum()
    print nerr_coded

def run_MAC_rand_chan_gain(SNR_vals, M, K, K2 = 1): 
  n = 64800
  kb = 48600
  ks = 43200
#  kb = 32400
#  ks = 32400
#  n = 6
#  kb = 3
#  ks = 3
  Nb = 2
  Ns = 1
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h = 1)

  Hs = LDPC_lib.LDPC(n, ks)
  Hs.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-ks)))
  Hs.prepare_decoder_C()
  bsA = np.random.randint(2,size=[M * Ns, ks])
  bsB = np.random.randint(2,size=[M * Ns, ks])
  csA = Hs.encode(bsA).flatten()
  csB = Hs.encode(bsB).flatten()

  Hb = LDPC_lib.LDPC(n, kb)
  Hb.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-kb)))
  Hb.prepare_decoder_C()  
  bbA =  np.random.randint(2,size=[M * Nb, kb])
  bbB =  np.random.randint(2,size=[M * Nb, kb])
  cbA  = Hb.encode(bbA).flatten()
  cbB  = Hb.encode(bbB).flatten()

  c_mapA = csA*4 + cbA[0::2]*2 + cbA[1::2] # fixed for Nb = 2, Ns = 1
  sA = sourceA_const[c_mapA]

  c_mapB = csB*4 + cbB[0::2]*2 + cbB[1::2] # fixed for Nb = 2, Ns = 1
  sB = sourceB_const[c_mapB]
 
  c_map = csA*8 + csB*4 + (cbA[0::2]  ^ cbB[0::2])*2 + (cbA[1::2]  ^ cbB[1::2])

#  hA = np.random.rayleigh(np.sqrt(2)/2, size = M * n) * np.exp(1j * np.random.uniform(0,2*np.pi, size = M * n))
#  hB = np.random.rayleigh(np.sqrt(2)/2, size = M * n) * np.exp(1j * np.random.uniform(0,2*np.pi, size = M * n))
  hA = np.ones(M * n)
  hB = np.ones(M * n)
  hB = np.exp(1j * np.random.uniform(0,2*np.pi/16., size = M * n))
  Pbe = []
  for SNR in SNR_vals:
    (x,sigma2w) = AWGN(hA * sA + hB * sB, SNR)
    x = x.flatten()
    muR = np.zeros([n * M, 2 ** (2*Nb + 2*Ns)], float)
    muH = np.zeros([n * M, 2 ** (Nb + 2*Ns)], float) 

    for i in range(1 << (Nb + Ns)):
      for j in range(1 << (Nb + Ns)):
        ind = i + (1 << (Nb + Ns)) * j
        muR[:, ind] = np.exp(-np.abs(x - hA * sourceA_const[i] - hB * sourceB_const[j])**2/sigma2w) 
        inds = np.hstack([dec2bin(i,3), dec2bin(j,3)])
        hier_ind = bin2dec(np.array([inds[0], inds[3], inds[1]^inds[4], inds[2]^inds[5]]))
        muH[:, hier_ind] += muR[:, ind]    
#    print np.max(muR,axis = 1), np.argmax(muR, axis = 1)
#    print np.max(muH, axis = 1), np.argmax(muH, axis = 1)
    if (np.min(muH) == 0):
      muH += 1e-100


    nerr = (np.argmax(muH, axis=1) != c_map).sum()
    print nerr
    est_csA, est_csB, est_cb  = BC_decode(muH, Hb, Hs, M, Nb, Ns, K, K2)
    est_c = est_csA*8 + est_csB*4 + est_cb

    nerr_coded = (est_c != c_map).sum()
    print nerr_coded
    Pbe.append(float(nerr_coded)/n/M)
  return Pbe

def run_Dest_dec_simple(gHSI, M, K, K2 = 1, alp = 0.8): 
  """Runs the decoder in destinations B only for HSI, basic parts and a priory knowledge about sAb XOR
  sBb (therefore simple). 
  
  Inputs: gHSI ... SNR of HSI link
          M    ... number of frames
          K    ... number of iterations within binary LDPC decoder
          K2   ... number of iterations in SoDeM
          alp  ... |hA| = |hB| * alp - relative strenght between channels
          """

  n = 64800 
#  kb = 51840 #4/5
  kb = 48600 #3/4
#  kb = 43200 #2/3
#  kb = 38880 #3/5
#  kb = 32400 #1/2
#  n = 6
#  kb = 3
  Nb = 2
  Ns = 1
  lenB = (1 << Nb)
  (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h = 1)
  
  H = LDPC_lib.LDPC(n, kb)
  H.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-kb)))
  H.prepare_decoder_C()  
  bA =  np.random.randint(2,size=[M * Nb, kb])
  cA  = H.encode(bA).flatten()
  bB =  np.random.randint(2,size=[M * Nb, kb])
  cB  = H.encode(bB).flatten()

  cA_map = np.asarray([cA[i::Nb]*(1 << (Nb - i - 1)) for i in range(Nb)]).sum(axis = 0)
  cB_map = np.asarray([cB[i::Nb]*(1 << (Nb - i - 1)) for i in range(Nb)]).sum(axis = 0)
  
  cAB = cA_map ^ cB_map
  c_map = cA_map * lenB + cB_map

#  hA = np.random.rayleigh(np.sqrt(2)/2, size = M * n) * np.exp(1j * np.random.uniform(0,2*np.pi, size = M * n))
#  hA = np.ones(M * n,float)
#  hB = alp * np.random.rayleigh(np.sqrt(2)/2, size = M * n) * np.exp(1j * np.random.uniform(0,2*np.pi, size = M * n))


  hA = 1
  hB = 1
  s = hA * basic_part[cA_map] + hB * basic_part[cB_map] * alp
  

  (x,sigma2w) = AWGN(s, gHSI)
  x = x.flatten()
  
  #    for i in const:
  #        plt.plot(np.real(i), np.imag(i), 'bx', ms=10)
  #    for i in x:
  #        plt.plot(np.real(i), np.imag(i), 'or', ms=5)
  #    plt.show()
  
 # muR = np.zeros([n * M, 2 ** (2*Nb)], float) # Joint metric
  #      muR[:, i * (1 << Nb) + j] = np.exp(-np.abs(x - hA * basic_part[i] - hB * basic_part[j])**2/sigma2w) 
  muR = np.zeros([n * M, lenB], float) 
  for k in range(lenB):
    act_range, = np.nonzero(k == cAB)
    const = np.asarray([hA * basic_part[i] + alp * hB * basic_part[i^k]  for i in range(lenB)])
    for i in range(lenB):
#      muR[act_range, i] = np.exp(-np.abs(x[act_range] - const[i, act_range])**2/sigma2w )
      muR[act_range, i] = np.exp(-np.abs(x[act_range] - const[i])**2/sigma2w )
  
  if (np.min(muR) == 0):
    muR += 1e-100
  
  est_cA = np.argmax(muR, axis=1)
  nerr = ((est_cA ^ cAB)  != cB_map).sum()
  print nerr # uncoded

  est_cAc = gen_decode_mod(muR, H, M, Nb, K, K2).flatten()
  est_cA_map = np.asarray([est_cAc[i::Nb]*(1 << (Nb - i - 1)) for i in range(Nb)]).sum(axis = 0)
  nerrC = ((est_cA_map ^ cAB) != cB_map).sum()
  print nerrC
  return float(nerrC) / M / H.n
#    est_csA, est_csB, est_cb  = BC_decode(muR, Hb, Hs, M, Nb, Ns, K, K2)
#    est_c = est_csA*8 + est_csB*4 + est_cb

#    est  = gen_decode(muR, [Hs,Hs,Hb,Hb], M, Nb+2*Ns, K, K2)
#    est_cc = est[:,0]*8 + est[:,1]*4 + est[:,2]*2 + est[:,3]
#    print est, est_cc, '\n', np.asarray([dec2bin(i,4) for i in est_c]), est_c,'\n', c_map

#    nerr_coded = (est_c != c_map).sum()
#    nerr_coded1 = (est_cc != c_map).sum()
#    print nerr_coded1 #, nerr_coded


def GF2_decode_test(M, K, SNR):
  n = 64800
  kb = 32400
#  kb = 48600
#  n = 6
#  kb = 3
  Nb = 1
  H = LDPC_lib.LDPC(n, kb)
  H.load_o('lib/LDPC_library/LDPC_{0}_{1}'.format(str(n),str(n-kb)))
  H.prepare_decoder_C()
  const = np.exp(1j * np.arange(0,2)*np.pi/1)
  bb =  np.random.randint(2,size=[M * Nb, kb])
  cb  = H.encode(bb).flatten()
  c_map = np.zeros(M * n, int)
  for i in range(Nb):
    c_map += cb[i::Nb]*(1 << (Nb - i - 1))
#  c_map = cb[0::Nb]*2 + cb[1::Nb]
  
  s = const[c_map]
#  print inter, deinter, c

  (x,sigma2w) = AWGN(s, float(SNR))
  x = x.flatten()
  mub = np.zeros([n * M, 1 << Nb], float) # 

  for i in range(0, 1 << Nb ):
    mub[:, i] = np.exp(-np.abs(x - const[i])**2/sigma2w) 

  if (np.min(mub) == 0):
    mub += 1e-100
   
  nerr = (np.argmax(mub, axis=1) != c_map).sum()
  print 'Uncoded nerr: ', nerr
  musb = np.zeros([n * M * Nb, 2])
  musb_o = np.zeros([n * M * Nb, 2])
  for i in range(1 << Nb):
    inds = dec2bin(i, Nb)
    for j in range(Nb):
      if inds[j] == 0:
        musb[j::Nb, 0] += mub[:, i]
      else:
        musb[j::Nb, 1] += mub[:, i]  
  est = musb[:,1] > musb[:,0]
  print 'Binary uncoded nerr: ', np.sum(np.abs(est - cb.flatten()))

#  print cb, musb, est
  LLR_vec = np.log(musb[:,1]/musb[:,0])
  dec = np.zeros(LLR_vec.shape,dtype=float)
  soft_out = np.zeros(LLR_vec.shape,dtype=float)
  t1 = np.zeros([M * Nb,H.num_edges],dtype=float).flatten()
  t2 = np.zeros([M * Nb,H.num_edges],dtype=float).flatten()
  _LDPC_dec_lib.run_FG_SPA_ef_soft_out_func(t1, t2, dec, -LLR_vec, soft_out, \
    H.FN_deg_distr, H.FN_map, H.FN_map_for_C, H.VN_deg_distr, H.VN_map_for_C, M * Nb, K, 1)
  soft_out[np.nonzero(np.isnan(soft_out))] = 0 # avoiding nans
  musb_o[:,1] =  H.LLR2pr1(soft_out)
  musb_o[:,0] =  1 - musb_o[:,1]
  mub_ot =  np.ones([M * H.n, 1 << Nb], float) 
  for i in range(1 << Nb):
    inds = dec2bin(i, Nb)
    for j in range(Nb):
      mub_ot[:, i] = mub_ot[:, i] * musb_o[j::Nb, inds[j]]
  mub_o = mub_ot/(np.array([mub_ot.sum(axis=1)]).repeat(1 << Nb,axis=0)).transpose() # norming

#  print LLR_vec, cb, c_map, soft_out

  nerr = (np.argmax(mub_o, axis=1) != c_map).sum()
  print 'Decoded nerr: ', nerr

  dec[np.nonzero(np.isnan(dec))] = 0
  est = (dec < 0)
#  print dec, est+1-1
  print 'Decoded Binary nerr: ', np.sum(est != cb.flatten()) #, np.sum(c_est != c.flatten())

      

#n = 64800
#k = 32400
#n = 6 
#k = 3
M = 1
#SNR_vals = np.arange(2.4,2.9,0.1)
SNR_vals = np.array([15])
K = 30

par = 0
#GF4_run(n, k, SNR_vals, M, K)
#GF2_run(n, k, SNR_vals, M, K, 15)
