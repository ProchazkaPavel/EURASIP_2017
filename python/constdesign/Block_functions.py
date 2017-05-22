import numpy as np
import matplotlib.pyplot as plt

def eval_sigma2w(SNR):
  gammaAdB = float(SNR)
  gammaA= 10.0**(gammaAdB/10)
  return 1.0 / (gammaA)


def dec2bin(x, num_bits=None): # Convert decimal number to binary narray
  return np.asarray([int(c)  for c in np.binary_repr(x,width=num_bits)],dtype=int) # dec to bin array

def dec2binvec(vec, num_bits):
  l = len(vec)
  bits = np.zeros([num_bits, l],int)
  for i in range(num_bits): # Reverting the symbols back to bits
    shift = (1 << i)
    bits[num_bits - 1 - i, :] = (vec & shift) >> i
  return bits.flatten(1)

def bin2dec(vec): # Convert binary narray to decimal number
  return int(np.array_str(vec).strip('[]').replace(' ',''),2)

def bin2decvec(vec, num_bits):
  l = len(vec)
  vecr = vec.reshape(l/num_bits, num_bits).transpose()
  d = np.zeros(l/num_bits, int)
  for i in range(num_bits): 
    d += vecr[num_bits - 1 - i, :] * (1 << i)
  return d

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

def const_design_XOR(Nb, Ns, phi = 0):
  (Lb, Ls) = eval_levels_XOR(Nb, Ns)
  en = np.sum(np.abs(Lb)**2) + np.sum(np.abs(Ls)**2)
  alpha = 1./np.sqrt(en)
  sourceA_const = find_levels(0, np.hstack([Ls, Lb]), [0], [-1., 1.])*alpha
  sourceB_const = find_levels(0, np.hstack([1j*Ls, Lb]), [0], [-1., 1.])*alpha
  basic_part = find_levels(0,Lb, [0], [-1., 1.])*alpha
  relay_const = np.asarray([x + y * np.exp(1j * phi) for x in sourceA_const for y in sourceB_const])
  return (sourceA_const, sourceB_const, basic_part, relay_const, alpha)

def const_superposed_XOR(Nb, Ns, phi = 0):
  (Lb, Ls) = eval_levels_XOR(Nb, Ns)
  en = np.sum(np.abs(Lb)**2) + np.sum(np.abs(Ls)**2)
  alpha = 1./np.sqrt(en)
  return find_levels(0, Ls, [0], [-1., 1.])*alpha
#  sourceA_const = find_levels(0, Ls, [0], [-1., 1.])*alpha
#  sourceB_const = find_levels(0, Ls, [0], [-1., 1.])*alpha * 1j
#  return np.asarray([x + y for x in sourceA_const for y in sourceB_const])

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
  return (Lb, Ls) # Basic and superposed levels

def const_design_MS(Nb, Ns, phi = 0):
  (Lb, Ls) = eval_levels_MS(Nb, Ns)
  en = np.sum(np.abs(Lb)**2) + np.sum(np.abs(Ls)**2)
  alpha = 1./np.sqrt(en)
  sourceA_const = find_levels(0, np.hstack([Ls, Lb]), [0], [-1., 1.])*alpha
  sourceB_const = find_levels(0, np.hstack([1j*Ls, Lb]), [0], [-1., 1.])*alpha
  basic_part = find_levels(0,Lb, [0], [-1., 1.])*alpha
  relay_const = np.asarray([x + y * np.exp(1j * phi)  for x in sourceA_const for y in sourceB_const])
  return (sourceA_const, sourceB_const, basic_part, relay_const, alpha)

def const_superposed_MS(Nb, Ns, phi = 0):
  (Lb, Ls) = eval_levels_MS(Nb, Ns)
  en = np.sum(np.abs(Lb)**2) + np.sum(np.abs(Ls)**2)
  alpha = 1./np.sqrt(en)
  return find_levels(0, Ls, [0], [-1., 1.])*alpha


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
  return const / scale

def Modulator_network(in0, Nb, Ns, node, mapping):
    if mapping == 'XOR':
      (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns)
    elif mapping == 'MS':
      (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns)
#    print 'sA=',sourceA_const,'\n', 'sB=',sourceB_const,'\n', 'basic=',basic_part,'\n'
    if node == 1: #sA
      return sourceA_const[in0]
    elif node == 2: #sB
      return sourceB_const[in0]

def Demodulator_relay(in0, Nb, Ns, mapping, hA=1, hB=1):
    if mapping == 'XOR':
      (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns)
      ind_const = np.asarray([s1* 2**(Nb + Ns) + s2 * 2**(Nb) + (b1^b2) \
              for s1 in range(0, 2**Ns) for b1 in range(0, 2**Nb) \
              for s2 in range(0, 2**Ns) for b2 in range(0, 2**Nb)] )
    elif mapping == 'MS':
      (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns)
      ind_const = np.asarray([s1* 2**(Nb + Ns) + s2 * 2**(Nb) + ((b1+b2) % (2**Nb)) \
              for s1 in range(0, 2**Ns) for b1 in range(0, 2**Nb) \
              for s2 in range(0, 2**Ns) for b2 in range(0, 2**Nb)] )
    dist = np.asarray([np.abs(in0 - hA * sA - hB * sB)**2 \
    for sA in sourceA_const for sB in sourceB_const] )
    return ind_const[np.argmin(dist, axis = 0)]

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

def Modulator_QAM(in0, N):
    const = QAM(N)
    return const[in0]

def Demodulator_QAM(N, in0, h=1):
    const = QAM(N)
    dist = np.asarray([np.abs(in0 - h*s0)**2 for s0 in const] )
    return  np.argmin(dist, axis = 0)

def Modulator_Superposed(dH, Nb, Ns, node, mapping): # Superposed part for the interference cancellation
    if mapping == 'XOR':
      const = const_superposed_XOR(Nb, Ns)
    if mapping == 'MS':
      const = const_superposed_MS(Nb, Ns)
    db = dH % (2**(Nb))
    dBs = (dH >> Nb)  % (2**Ns)
    dAs = (dH >> (Nb + Ns))
    if node == 1: # DA
      return const[dBs] * 1j
    elif node == 2: #DB
      return const[dAs]

def Demodulator_Basic(in0, Nb, Ns, mapping, h=1):
    if mapping == 'XOR':
      (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_XOR(Nb, Ns)
    elif mapping == 'MS':
      (sourceA_const, sourceB_const, basic_part, relay_const, alpha) = const_design_MS(Nb, Ns)
    const = basic_part
    dist = np.asarray([np.abs(in0 - h*s0)**2 for s0 in const] )
    return np.argmin(dist, axis = 0)


def Eval_Data(Nb, Ns, node, mapping, dH, d):
    dbh = dH % (2**(Nb)) # NCed basic parts
    if mapping ==  'XOR':
      db = d ^ dbh # basic part
    elif mapping ==  'MS':
      db = ((dbh - d) % (2**Nb)) # basic part
    dBs = (dH >> Nb)  % (2**Ns)
    dAs = (dH >> (Nb + Ns))

    if node == 1: # DA
      return db + (dAs << Nb)
    elif node == 2: # DB
      return db + (dBs << Nb)

def Error_Overall_Check(d, d_est):
    nerr = len(np.nonzero(d != d_est)[0])
    return (d != d_est).sum()/float(len(d))
