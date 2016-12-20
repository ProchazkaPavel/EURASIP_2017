import numpy as np
from matplotlib.patches import Polygon, Circle
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt


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
    superposed_part = find_levels(0,Ls, [0], [-1., 1.])*alpha
    relay_const = np.asarray([x + y * h for x in sourceA_const for y in sourceB_const])
    return (sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha)

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

def SNR2sigma2w(SNR):
    alpha = 10**(float(SNR)/10)
    return 1.0/alpha

#plt.plot(np.real(relay_const), np.imag(relay_const), 'ok', ms=10)


#%%
def intersect(lin, lin_vec):
    '''
    intersection of lines
    '''
    [a, b, c] = lin
    res = []
    for L2 in lin_vec:
        [a1, b1, c1] = L2        
        res.append((b*c1 - b1*c)/(a*b1 - a1*b)+ 1j*(-a*c1 + a1*c)/(a*b1 - a1*b))
    return np.asarray(res)
        


def prepare_lines(Points):
    '''
    Evaluate lines perpendicular to the vector given by individual points
    line is given by ax + by + c = 0
    '''
    lins = []
    for p in Points:
        if np.abs(np.imag(p)) < 1e-10:
            py = np.inf # no crossing y (imag) axis        
            px = np.real(p)+np.imag(p)**2/np.real(p) # cross x axis
            c = px
            a = -1
            b = 0        
        elif np.abs(np.real(p)) < 1e-10:
            px = np.inf # no crossing x (real) axis     
            py = np.abs(p)/np.sin(np.angle(p))*1j # cross y axis
            a = 0  
            b = -1
            c = float(np.imag(py))
        else:
            px = np.real(p)+np.imag(p)**2/np.real(p) # cross x axis
            py = np.abs(p)/np.sin(np.angle(p))*1j # cross y axis
            c = np.imag(py)
            b = -1
            a = -c/np.real(px)
            
        lins.append([a,b,c])
    return np.asarray(lins)

def find_border_points(lins, Points):
    '''
    Starting with closest perpendicular point as reference point, 
    selecting the closest crossing of lines as the next refernce point
    '''
    ind_sel = np.argmin(np.abs(Points)) # the first selected point (index)
    selected = [ind_sel]    
    sel_intersect = []
    cnt = 0
    ref_point = Points[ind_sel]
    while True:        
        point_sel = Points[ind_sel] 
        lin_sel = lins[ind_sel]      
        
        P_rem = Points * np.exp(-1j*np.angle(point_sel)) # rotation that selected point has angle zero    
        sat_inds, = np.nonzero((np.mod(np.angle(P_rem), 2*np.pi) > 1e-10)*(np.mod(np.angle(P_rem), 2*np.pi) < np.pi - 1e-10)) # positive indecis    
        intersects = intersect(lin_sel, lins[sat_inds])    
        dists = np.abs(intersects - ref_point)        
        ind_min = np.argmin(dists)
        sel_intersect.append(intersects[ind_min])
        ref_point = intersects[ind_min]
        ind_sel = sat_inds[ind_min]
        cnt+=1
        if ind_sel in selected: # loop closed
            break
        else:
            selected.append(ind_sel)       
    return np.hstack([sel_intersect, sel_intersect[0]])

def unique_complex(vec, eq = 1e-5):
    '''
    unique the vector, where equality means np.abs(a-b)<eq
    '''
    uniques_vec = [vec[0]]
    mapping = [0]
    for i, c in enumerate(vec[1:]):
        if np.min(np.abs(c-np.asarray(uniques_vec))) > eq:
            uniques_vec.append(c) 
            mapping.append(i+1)
    return np.asarray(uniques_vec), np.asarray(mapping)

def check_excl_failures(relay_const, Nb, Ns, eq = 1e-5):
    '''
    unique the vector, where equality means np.abs(a-b)<eq and check for the exlusive law failures
    '''
    uniques_vec = []
    mapping = []
    J2H = np.zeros(4**(Nb+Ns))
    C2H = [] # mapped symbols to hier symbol
    C2B = [] # mapped symbols to basic part
    Failures = []    
    for dAb in range(2**Nb):
        for dBb in range(2**Nb):
            for dAs in range(2**Ns):
                for dBs in range(2**Ns):
                    ind = dAs*2**(Ns+2*Nb) + dAb*2**(Ns+Nb)+dBs*2**(Nb) + dBb
                    ind_b = dAb ^ dBb
                    indH = dAs*2**(Ns+Nb) + dBs*2**(Nb) + ind_b
                    c = relay_const[ind]
                    if (len(uniques_vec) == 0) or (np.min(np.abs(c-np.asarray(uniques_vec))) > eq):
                        uniques_vec.append(c)
                        mapping.append(J2H)
                        C2H.append(indH)
                        C2B.append(ind_b)
                        Failures.append(False)                        
                    else:
                        for pos in np.flatnonzero(np.abs(c-np.asarray(uniques_vec)) <= eq):
                            if C2H[pos] != indH:
                                Failures[pos] = True
    return C2H, C2B, np.asarray(uniques_vec), Failures
    

def Get_Dec_Region_Boundaries(const, mmax = 3):
    '''
    Return list of borders defining polygons for decision region
    '''
    bord = []
    unique_const, mapping = unique_complex(const)
    for ind, c in enumerate(unique_const):        
        bord_frame = np.array([mmax - np.real(c),-mmax - np.real(c), 1j*mmax - 1j*np.imag(c),-1j*mmax - 1j*np.imag(c)])
        _Points = (np.hstack([unique_const[:ind], unique_const[(ind+1):]]) - c)*0.5
        Points = np.hstack([bord_frame, _Points])    
        lins = prepare_lines(Points)
        bord.append(find_border_points(lins, Points) + c)
    return bord
    
def Draw_dec_Boundaries(ax, bords, hatches, mapping = None, Failures = None):
    '''
    Draw the boundaries
    '''       
    for i,  bord in enumerate(bords):
        #ax.plot(np.real(bord), np.imag(bord),'k-', lw=2)   
        xy=np.vstack([np.real(bord), np.imag(bord)]).T
        if mapping is not None:
            if not Failures[i]:
                bar = plt.Polygon(xy, ec='k',lw=2, hatch=hatches[mapping[i]], facecolor='white')
            else:
                bar = plt.Polygon(xy, ec='k',lw=2, facecolor='red', fill=True, alpha=0.4)
            ax.add_artist(bar)      
        else:
            bar = plt.Polygon(xy, facecolor='white', ec='k',lw=2, label=str(i))
    #ax.plot(np.real(const), np.imag(const),'ok',ms=8)
    
def Draw_Error_Circles(ax, const, SNR, mapping=None, Pe = 1e-5):
    '''
    Draw the circles for the given constellation
    '''
    sigma2w = SNR2sigma2w(SNR)
    d = np.sqrt(-sigma2w*np.log(Pe))
    if mapping is None:
        for p in const:
            min_dist = np.min(np.asarray([np.abs(p1 - p) for p1 in const if  np.abs(p1-p)>1e-10]))
            if min_dist < 2*d:
                circ = plt.Circle((np.real(p), np.imag(p)), d, color='red',fill=None,lw = 4)
            else:
                circ = plt.Circle((np.real(p), np.imag(p)), d, color='green',fill=None,lw = 4)
            ax.add_artist(circ)
    else:
        for i, p in enumerate(const):
            min_dist = np.min(np.asarray([np.abs(p1 - p) \
                        for i1, p1 in enumerate(const) \
                        if  (np.abs(p1-p)>1e-10) and (mapping[i] != mapping[i1])]))            
            if min_dist < 2*d:
                circ = plt.Circle((np.real(p), np.imag(p)), d, color='red',fill=None,lw = 4)
            else:
                circ = plt.Circle((np.real(p), np.imag(p)), d, color='green',fill=None,lw = 4)
            ax.add_artist(circ)






def Draw_Relay_BC(ax, Nb, Ns):
    const = QAM(Nb+2*Ns)[0]
    base_hatch_Relay = ['o', 'O','.','*', '\\']
    hatches_Relay = [''.join([base_hatch_Relay[int(i)] for i in np.base_repr(ind,len(base_hatch_Relay))]) for ind in range(2**Nb)]
    col = cm.rainbow(np.linspace(0,1,2**Ns))
    ax.grid()
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.axis('equal')
    for db in range(2**Nb):
        for dAs in range(2**Ns):
            for dBs in range(2**Ns):                                        
                indH = dAs*2**(Ns+Nb) + dBs*2**(Nb) + db
                c = const[indH]
                ax.scatter(np.real(c), np.imag(c), s=1000, marker='s', hatch=hatches_Relay[db], facecolor='white')                
                ax.plot(np.real(c), np.imag(c), 'o', color=col[dAs], mew=3, ms=16, fillstyle='left')
                ax.plot(np.real(c), np.imag(c), 'o', color=col[dBs], mew=3, ms=16, fillstyle='right')                    
    
def Draw_Destination_BC(ax, Nb, Ns):
    const = QAM(Nb+2*Ns)[0]
    base_hatch_Relay = ['o', 'O','.','*', '\\']
    hatches_Relay = [''.join([base_hatch_Relay[int(i)] for i in np.base_repr(ind,len(base_hatch_Relay))]) for ind in range(2**Nb)]
    col = cm.rainbow(np.linspace(0,1,2**Ns))
    ax.grid()
    ax.axis('equal')
    C2B = np.zeros(len(const), int)
    for db in range(2**Nb):
        for dAs in range(2**Ns):
            for dBs in range(2**Ns):                                        
                indH = dAs*2**(Ns+Nb) + dBs*2**(Nb) + db
                C2B[indH] = db
                
    bords = Get_Dec_Region_Boundaries(const, 1.5) #prepare decision regions for relay constellatio
    Draw_dec_Boundaries(ax, bords, hatches_Relay, C2B, np.zeros(len(const), bool))
    for db in range(2**Nb):
        for dAs in range(2**Ns):
            for dBs in range(2**Ns):                                        
                indH = dAs*2**(Ns+Nb) + dBs*2**(Nb) + db
                c = const[indH]
                ax.plot(np.real(c), np.imag(c), 'o', color=col[dAs], mew=3, ms=16, fillstyle='left')
                ax.plot(np.real(c), np.imag(c), 'o', color=col[dBs], mew=3, ms=16, fillstyle='right')                    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

def Draw_Relay_Const(ax, Nb, Ns, h, mmax = 3.):
    (sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)    
    col = cm.rainbow(np.linspace(0,1,2**Ns))
    base_hatch_Relay = ['o', 'O','.','*', '\\']
    hatches_Relay = [''.join([base_hatch_Relay[int(i)] for i in np.base_repr(ind,len(base_hatch_Relay))]) for ind in range(2**Nb)]
    ax.axis('equal')
    #ax.set_xlim([-mmax,mmax])
    #ax.set_ylim([-mmax,mmax])
    ax.grid()    
    C2H, C2B, unique_const, Failures = check_excl_failures(relay_const, Nb, Ns, eq = 1e-5)
    bords = Get_Dec_Region_Boundaries(unique_const, mmax) #prepare decision regions for relay constellation
    Draw_dec_Boundaries(ax, bords, hatches_Relay, C2B, Failures)    
    
    col = cm.rainbow(np.linspace(0,1,2**Ns)) 
    for dAb in range(2**Nb):
        for dBb in range(2**Nb):
            for dAs in range(2**Ns):
                for dBs in range(2**Ns):                                        
                    c = basic_part[dAb] + h*basic_part[dBb] + superposed_part[dAs] + h*1j*superposed_part[dBs]
                    ax.plot(np.real(c), np.imag(c), 'o', color=col[dAs], mew=3, ms=16, fillstyle='left')
                    ax.plot(np.real(c), np.imag(c), 'o', color=col[dBs], mew=3, ms=16, fillstyle='right')                    
    
def Draw_SRC_Const(ax, Nb, Ns, node = 'A'):
    (sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha) =    const_design_XOR(Nb, Ns, 1.)    
    col = cm.rainbow(np.linspace(0,1,2**Ns))
    base_hatch_SRC = ['/','-', '+', 'x']
    hatches_SRC = [''.join([base_hatch_SRC[int(i)] for i in np.base_repr(ind,len(base_hatch_SRC))]) for ind in range(2**Nb)]
 
    ax.axis('equal')
    ax.set_xlim([-1.5, 1.5])
    #ax.set_ylim([-1.5, 1.5])
    ax.grid()
    
    for db in range(2**Nb):
        for ds in range(2**Ns):
            sA = basic_part[db] + superposed_part[ds]
            sB = basic_part[db] + 1j*superposed_part[ds]
            if node == 'A':                               
                ax.plot(np.real(sA), np.imag(sA), 'o', color=col[ds], mew=3, ms=16, fillstyle='left')                                
                ax.scatter(np.real(sA), np.imag(sA), s=1000, marker='s', hatch=hatches_SRC[db], facecolor='white')                
            else:
                ax.plot(np.real(sB), np.imag(sB), 'o', color=col[ds], mew=3, ms=16, fillstyle='right')
                ax.scatter(np.real(sB), np.imag(sB), s=1000, marker='s', hatch=hatches_SRC[db], facecolor='white')                

def Draw_Received_Dest_MAC(ax, Nb, Ns, node='A'):
    (sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha) = const_design_XOR(Nb, Ns, 1.)    
    col = cm.rainbow(np.linspace(0,1,2**Ns))
    base_hatch_SRC = ['/','-', '+', 'x']
    hatches_SRC = [''.join([base_hatch_SRC[int(i)] for i in np.base_repr(ind,len(base_hatch_SRC))]) for ind in range(2**Nb)]
    C2B = np.zeros(len(sourceA_const), int)
    col = cm.rainbow(np.linspace(0,1,2**Ns)) 
    if node == 'B':
        bords = Get_Dec_Region_Boundaries(sourceA_const, 1.5) #prepare decision regions for relay constellation
        for b in range(2**Nb):
            for s in range(2**Ns):
                indA = s * 2**Nb + b 
                C2B[indA] = b 
        Draw_dec_Boundaries(ax, bords, hatches_SRC, C2B, np.zeros(len(sourceA_const), bool))    
    
        for b in range(2**Nb):
            for s in range(2**Ns):
                c = basic_part[b] +  superposed_part[s] 
                ax.plot(np.real(c), np.imag(c), 'o', color=col[s], mew=3, ms=16, fillstyle='left')
    elif node == 'A':
        bords = Get_Dec_Region_Boundaries(sourceB_const, 1.5) #prepare decision regions for relay constellation
        for b in range(2**Nb):
            for s in range(2**Ns):
                indB = s * 2**Nb + b 
                C2B[indB] = b 
        Draw_dec_Boundaries(ax, bords, hatches_SRC, C2B, np.zeros(len(sourceA_const), bool))    
    
        for b in range(2**Nb):
            for s in range(2**Ns):
                c = basic_part[b] +  1j*superposed_part[s] 
                ax.plot(np.real(c), np.imag(c), 'o', color=col[s], mew=3, ms=16, fillstyle='right')                    
    ax.grid()    
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])

def Draw_Received_Dest_MAC_IC(ax, Nb, Ns):
    (sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha) = const_design_XOR(Nb, Ns, 1.)    
    base_hatch_SRC = ['/','-', '+', 'x']
    hatches_SRC = [''.join([base_hatch_SRC[int(i)] for i in np.base_repr(ind,len(base_hatch_SRC))]) for ind in range(2**Nb)]
    C2B = np.arange(len(basic_part))
    bords = Get_Dec_Region_Boundaries(basic_part, 1.5) #prepare decision regions for relay constellation
    ax.axis('equal')
    ax.set_xlim([-1.5,1.5])
    ax.grid()    
    #ax.set_ylim([-1.5,1.5])
    Draw_dec_Boundaries(ax, bords, hatches_SRC, C2B, np.zeros(len(sourceA_const), bool))    

    ax.plot(np.real(basic_part), np.imag(basic_part), 'o', color='k', mew=3, ms=16, fillstyle='none')

def insert_legend(Nb, Ns, ax):
    col = cm.rainbow(np.linspace(0,1,2**Ns))
    base_hatch_SRC = ['/','-', '+', 'x']
    base_hatch_Relay = ['o', 'O','.','*', '\\']
    hatches_SRC = [''.join([base_hatch_SRC[int(i)] for i in np.base_repr(ind,len(base_hatch_SRC))]) for ind in range(2**Nb)]
    hatches_Relay = [''.join([base_hatch_Relay[int(i)] for i in np.base_repr(ind,len(base_hatch_Relay))]) for ind in range(2**Nb)]

    for i in range(2**Ns):
        Bleg, = ax.plot(-100,-100,'o', color=col[i], mew=3, ms=16, fillstyle='right', label='$d_B^s$=%d'%i)
        Aleg, = ax.plot(-100,-100,'o', color=col[i], mew=3, ms=16, fillstyle='left', label='$d_A^s$=%d'%i)
    [ax.scatter(-100, -100 , s=500, marker='s', facecolor='white', hatch=hatches_SRC[i], label=r'$d^b_S$=%d'%i) for i in range(2**Nb)]
    [ax.scatter(-100, -100 , s=500, marker='s', facecolor='white', hatch=hatches_Relay[i], label='$d^b_R$=%d'%i) for i in range(2**Nb)]
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.legend(scatterpoints=1, numpoints=1,ncol=2)
    ax.axis('off')

    
#%%
def draw_Constellations_S2R(Nb, Ns, h):
    '''
    Draw constellations in sources and corresponding received constellation in relay for given
    relative fading coefficient h
    '''
    fig = plt.figure(figsize=(16,10))

    ax_dA = plt.subplot2grid((2,4), (0, 0));
    Draw_SRC_Const(ax_dA, Nb, Ns, node = 'A')
    ax_dA.set_title('Source A constellation - MAC')

    ax_dB = plt.subplot2grid((2,4), (1, 0));
    Draw_SRC_Const(ax_dB, Nb, Ns, node = 'B')
    ax_dB.set_title('Source B constellation - MAC')

    ax_R = plt.subplot2grid((2,4), (0, 1), colspan=2, rowspan=2)
    Draw_Relay_Const(ax_R, Nb, Ns, h)
    ax_R.set_title('Received Relay constellation - MAC')

    ax_legend = plt.subplot2grid((2,4), (0, 3), rowspan=2)
    insert_legend(Nb, Ns, ax_legend)

def draw_Constellations(Nb, Ns, h):
    fig = plt.figure(figsize=(16,10))

    ax_SA = plt.subplot2grid((3,4), (0, 0));
    Draw_SRC_Const(ax_SA, Nb, Ns, node = 'A')
    ax_SA.set_title('Source A constellation - MAC')

    ax_SB = plt.subplot2grid((3,4), (1, 0));
    Draw_SRC_Const(ax_SB, Nb, Ns, node = 'B')
    ax_SB.set_title('Source B constellation - MAC')

    ax_R = plt.subplot2grid((3,4), (0, 1), colspan=2, rowspan=2)
    Draw_Relay_Const(ax_R, Nb, Ns, h)
    ax_R.set_title('Received Relay constellation - MAC')

    ax_DA = plt.subplot2grid((3,4), (0, 3));
    Draw_Received_Dest_MAC(ax_DA, Nb, Ns, node = 'A')
    ax_DA.set_title('Received $D_A$ - MAC')

    ax_DB = plt.subplot2grid((3,4), (1, 3));
    Draw_Received_Dest_MAC(ax_DB, Nb, Ns, node = 'B')
    ax_DB.set_title('Received $D_B$ - MAC')

    ax_legend = plt.subplot2grid((3,4), (2, 0))
    insert_legend(Nb, Ns, ax_legend)

    ax_R_BC = plt.subplot2grid((3,4), (2, 1));
    Draw_Relay_BC(ax_R_BC, Nb, Ns)
    ax_R_BC.set_title('Relay Constellation - BC')

    ax_DA_BC = plt.subplot2grid((3,4), (2, 2));
    Draw_Destination_BC(ax_DA_BC, Nb, Ns)
    ax_DA_BC.set_title('Received $D_A$ - BC')

    ax_DB_BC = plt.subplot2grid((3,4), (2, 3));
    Draw_Destination_BC(ax_DB_BC, Nb, Ns)
    ax_DB_BC.set_title('Received $D_B$ - BC')


def draw_Constellations_IC(Nb, Ns, h):
    fig = plt.figure(figsize=(16,10))

    ax_SA = plt.subplot2grid((3,4), (0, 0));
    Draw_SRC_Const(ax_SA, Nb, Ns, node = 'A')
    ax_SA.set_title('Source A constellation - MAC')

    ax_SB = plt.subplot2grid((3,4), (1, 0));
    Draw_SRC_Const(ax_SB, Nb, Ns, node = 'B')
    ax_SB.set_title('Source B constellation - MAC')

    ax_R = plt.subplot2grid((3,4), (0, 1), colspan=2, rowspan=2)
    Draw_Relay_Const(ax_R, Nb, Ns, h)
    ax_R.set_title('Received Relay constellation - MAC')

    ax_DA = plt.subplot2grid((3,4), (0, 3));
    Draw_Received_Dest_MAC_IC(ax_DA, Nb, Ns)
    ax_DA.set_title('Received $D_A$ - MAC-IC')

    ax_DB = plt.subplot2grid((3,4), (1, 3));
    Draw_Received_Dest_MAC_IC(ax_DB, Nb, Ns)
    ax_DB.set_title('Received $D_B$ - MAC-IC')

    ax_legend = plt.subplot2grid((3,4), (2, 0))
    insert_legend(Nb, Ns, ax_legend)

    ax_R_BC = plt.subplot2grid((3,4), (2, 1));
    Draw_Relay_BC(ax_R_BC, Nb, Ns)
    ax_R_BC.set_title('Relay Constellation - BC')

    ax_DA_BC = plt.subplot2grid((3,4), (2, 2));
    Draw_Destination_BC(ax_DA_BC, Nb, Ns)
    ax_DA_BC.set_title('Received $D_A$ - BC')

    ax_DB_BC = plt.subplot2grid((3,4), (2, 3));
    Draw_Destination_BC(ax_DB_BC, Nb, Ns)
    ax_DB_BC.set_title('Received $D_B$ - BC')

def draw_Constellations_IC_withErrCirc(Nb, Ns, h, gMAC, gBC, gHSI):
    fig = plt.figure(figsize=(16,10))
    (sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)    

    ax_SA = plt.subplot2grid((3,4), (0, 0));
    Draw_SRC_Const(ax_SA, Nb, Ns, node = 'A')
    ax_SA.set_title('Source A constellation - MAC')

    ax_SB = plt.subplot2grid((3,4), (1, 0));
    Draw_SRC_Const(ax_SB, Nb, Ns, node = 'B')
    ax_SB.set_title('Source B constellation - MAC')

    ax_R = plt.subplot2grid((3,4), (0, 1), colspan=2, rowspan=2)
    Draw_Relay_Const(ax_R, Nb, Ns, h)
    C2H, C2B, unique_const, Failures = check_excl_failures(relay_const, Nb, Ns, eq = 1e-5)
    Draw_Error_Circles(ax_R, unique_const, gMAC, C2H)
    ax_R.set_title('Received Relay constellation - MAC')

    ax_DA = plt.subplot2grid((3,4), (0, 3));
    Draw_Received_Dest_MAC_IC(ax_DA, Nb, Ns)
    Draw_Error_Circles(ax_DA, basic_part, gHSI)
    ax_DA.set_title('Received $D_A$ - MAC-IC')

    ax_DB = plt.subplot2grid((3,4), (1, 3));
    Draw_Received_Dest_MAC_IC(ax_DB, Nb, Ns)
    Draw_Error_Circles(ax_DB, basic_part, gHSI)
    ax_DB.set_title('Received $D_B$ - MAC-IC')

    ax_legend = plt.subplot2grid((3,4), (2, 0))
    insert_legend(Nb, Ns, ax_legend)

    ax_R_BC = plt.subplot2grid((3,4), (2, 1));
    Draw_Relay_BC(ax_R_BC, Nb, Ns)
    ax_R_BC.set_title('Relay Constellation - BC')

    ax_DA_BC = plt.subplot2grid((3,4), (2, 2));
    Draw_Destination_BC(ax_DA_BC, Nb, Ns)
    ax_DA_BC.set_title('Received $D_A$ - BC')
    Draw_Error_Circles(ax_DA_BC, QAM(Nb+2*Ns)[0], gBC)

    ax_DB_BC = plt.subplot2grid((3,4), (2, 3));
    Draw_Destination_BC(ax_DB_BC, Nb, Ns)
    ax_DB_BC.set_title('Received $D_B$ - BC')
    Draw_Error_Circles(ax_DB_BC, QAM(Nb+2*Ns)[0], gBC)




if __name__ == '__main__':
    Nb = 2
    Ns = 0
    h  = 0.96+0.12j
    mmax = 3

    (sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)

    col = cm.rainbow(np.linspace(0,1,len(relay_const))) 
    base_hatch = ['/','-', '+', 'x', 'o', 'O','.','*', '\\']
    hatches = [''.join([base_hatch[int(i)] for i in np.base_repr(ind,len(base_hatch))]) for ind in range(2**Nb)]
#hatches = [''.join([b for i in range(j)]) for b in base_hatch for j in range(2**Nb)]
               
       


    fig = plt.figure(figsize=(15,10))
    ax_dA = plt.subplot2grid((2,3), (0, 0));ax_dA.axis('equal')
    ax_dB = plt.subplot2grid((2,3), (1, 0));ax_dB.axis('equal')
    ax_R = plt.subplot2grid((2,3), (0, 1), colspan=2, rowspan=2)

    Draw_SRC_Const(ax_dA, Nb, Ns, node = 'A')
    Draw_SRC_Const(ax_dB, Nb, Ns, node = 'B')

    Draw_Relay_Const(ax_R, Nb, Ns, h)    





#%% DEBUG
#
#def Merge_2Borders(bord1, bord2):
#    '''
#    Merge selected 2 boundaries (if possible)
#    '''
#    inter =  np.intersect1d(bord1, bord2)    
#    if len(inter) == 2:        
#        ind1 = [np.flatnonzero(bord1 == i) for i in inter]
#        ind2 = [np.flatnonzero(bord2 == i) for i in inter]
#        res[0] = inter[0]
#        act_ind = ind1[0]      
#    else:   
#        return [bord1, bord2]
#
##%%
#import sympy as sp
#
#
#a,b,c,a1,b1,c1,x0,y0,x1,y2 = sp.symbols('a,b,c,a1,b1,c1,x0,y0,x1,y2')
#sp.solve([a*x0+b*y0+c, a1*x0+b1*y0+c1], x0, y0)
#
#
#
##%%
#ind = 0
#act_point = relay_const[ind]
#cc = np.hstack([relay_const[:ind],relay_const[(ind+1):]]) - act_point
#plt.plot(np.real(cc), np.imag(cc), 'ok', ms=10)
#plt.plot(0,0, 'og', ms=10)
#
#points = 0.5 * np.hstack([cc[:ind],cc[(ind+1):]]) # mid  points
#
#
#plt.plot(np.real(points), np.imag(points), 'xg', ms=5)
#
##%% preparation of lines
#plt.figure(figsize(12,12))
#c1=[]
#c2=[]
#mp = []
#lins = []
#N = 0
#plt.plot(0,0,'or', ms=15)
#Points = np.hstack([-1.5,1.5,-1.5j,1.5j, np.exp(1j*np.random.rand(N)*2*np.pi)*(1+0.1*np.random.rand(N))])
#
#for p in Points:
#    px = np.real(p)+np.imag(p)**2/np.real(p) # cross x axis
#    py = np.abs(p)/np.sin(np.angle(p))*1j # cross y axis
#    c = np.imag(py)
#    a = -c/np.real(px)
#    b = -1
#    if np.abs(np.imag(p)) < 1e-10:
#        py = np.inf # no crossing y (imag) axis        
#        c = px
#        a = -1
#        b = 0        
#    if np.abs(np.real(p)) < 1e-10:
#        px = np.inf # no crossing x (real) axis     
#        a = 0  
#        c = float(np.imag(py))
##    # check
##    c1.append(np.abs(px)**2 - np.abs(p-px)**2 - np.abs(p)**2)
##    c2.append(np.abs(p)**2+np.abs(py-p)**2-np.abs(py)**2)
#    lins.append([a,b,c])
#    plt.plot(np.real(p), np.imag(p), 'og', ms=5)
#    ran = np.array([-2,2])
#    if b == 0:
#        #plt.plot([c, c],ran, '--k')
#        mp.append(np.array([c, c])+ran*1j)
#    else:
#        #plt.plot(ran,a*ran+c, '--k')
#        mp.append(ran+(a*ran+c)*1j)
#
#lins = np.asarray(lins)
#   
#for i in mp:
#    plt.plot(np.real(i), np.imag(i), '--k')
#    
#
#selected = []
#sel_intersect = []
#bord_lines = []
#
#ind_sel = np.argmin(np.abs(Points)) # the first selected point (index)
#selected.append(ind_sel)
#
#while True:
#    point_sel = Points[ind_sel] 
#    lin_sel = lins[ind_sel]
#    
#    P_rem = Points * np.exp(-1j*np.angle(point_sel)) # rotation that selected point has angle zero    
#    sat_inds, = np.nonzero((np.mod(np.angle(P_rem), 2*np.pi) > 1e-10)*(np.mod(np.angle(P_rem), 2*np.pi) < np.pi - 1e-10)) # positive indecis    
#    intersects = intersect(lin_sel, lins[sat_inds])    
#    print point_sel, intersects, sat_inds
#    dists = np.abs(intersects - point_sel)
#    ind_min = np.argmin(dists)
#    sel_intersect.append(intersects[ind_min])
#    ind_sel = sat_inds[ind_min]
#    if ind_sel == selected[0]:
#        break
#    else:
#        selected.append(ind_sel)       
#borders = np.hstack([sel_intersect, sel_intersect[0]])
#
#
#plt.plot(np.real(borders), np.imag(borders), 'k-', lw=2)
