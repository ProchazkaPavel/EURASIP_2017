import numpy as np


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
    '''
    lins = []
    for p in Points:
        px = np.real(p)+np.imag(p)**2/np.real(p) # cross x axis
        py = np.abs(p)/np.sin(np.angle(p))*1j # cross y axis
        c = np.imag(py)
        a = -c/np.real(px)
        b = -1
        if np.abs(np.imag(p)) < 1e-10:
            py = np.inf # no crossing y (imag) axis        
            c = px
            a = -1
            b = 0        
        if np.abs(np.real(p)) < 1e-10:
            px = np.inf # no crossing x (real) axis     
            a = 0  
            c = float(np.imag(py))
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
    for c in vec[1:]:
        if np.min(np.abs(c-np.asarray(uniques_vec))) > eq:
            uniques_vec.append(c)
    return np.asarray(uniques_vec)

def Get_Dec_Region_Boundaries(const, mmax = 3):
    '''
    Return list of borders defining polygons for decision region
    '''
    bord = []
    unique_const = unique_complex(const)
    for ind, c in enumerate(unique_const):        
        bord_frame = np.array([mmax - np.real(c),-mmax - np.real(c), 1j*mmax - 1j*np.imag(c),-1j*mmax - 1j*np.imag(c)])
        _Points = (np.hstack([unique_const[:ind], unique_const[(ind+1):]]) - c)*0.5
        Points = np.hstack([bord_frame, _Points])    
        lins = prepare_lines(Points)
        bord.append(find_border_points(lins, Points) + c)
    return bord
    
def Draw_dec_Boundaries(ax, bords, col):
    '''
    Draw the boundaries
    '''        
    for i,  bord in enumerate(bords):
        ax.plot(np.real(bord), np.imag(bord),'k-', lw=2)    
        #xy=np.vstack([np.real(bord), np.imag(bord)]).T
        #bar = plt.Polygon(xy, color=col[i], ec='k',lw=2, label=str(i))
        #ax.add_artist(bar)        
    #ax.plot(np.real(const), np.imag(const),'ok',ms=8)
    

def Draw_Relay_Const(ax, Nb, Ns, h):
    (sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)    
    #markers_SRC = [(2+i, 1 , 0) for i in range(2**Nb)]
    markers_R = [(2+i, 2 , 0) for i in range(2**Nb)]
    col = cm.rainbow(np.linspace(0,1,2**(2*Ns)))
    for dAb in range(2**Nb):
        for dBb in range(2**Nb):
            for dAs in range(2**Ns):
                for dBs in range(2**Ns):
                    ind_s = dAs * 2**Ns + dBs
                    ind_b = dAb ^ dBb
                    c = basic_part[dAb] + h*basic_part[dBb] + superposed_part[dAs] + h*1j*superposed_part[dBs]
                    ax.plot(np.real(c), np.imag(c), marker=markers_R[ind_b], color=col[ind_s], mew=3, ms=16)
    
def Draw_SRC_Const(ax, Nb, Ns, node == 'A'):
    (sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)    
    markers_SRC = [(2+i, 1 , 0) for i in range(2**Nb)]
    #markers_R = [(2+i, 2 , 0) for i in range(2**Nb)]
    col = cm.rainbow(np.linspace(0,1,2**(2*Ns)))
    for db in range(2**Nb):
        for ds in range(2**Nb):
            ind_s = dAs * 2**Ns + dBs
            ind_b = dAb ^ dBb
            c = basic_part[dAb] + h*basic_part[dBb] + superposed_part[dAs] + h*1j*superposed_part[dBs]
            ax.plot(np.real(c), np.imag(c), marker=markers_R[ind_b], color=col[ind_s], mew=3, ms=16)

    
#%%
from matplotlib.patches import Polygon, Circle
from matplotlib.pyplot import cm
    
Nb = 2
Ns = 2
h  = 1.
mmax = 3

(sourceA_const, sourceB_const, basic_part, superposed_part, relay_const, alpha) = const_design_XOR(Nb, Ns, h)
col = cm.rainbow(np.linspace(0,1,len(const)))    
bords = Get_Dec_Region_Boundaries(relay_const, mmax)

fig, ax = plt.subplots()
Draw_Relay_Const(ax, Nb, Ns, h)    
Draw_dec_Boundaries(ax, bords, col)




#%% DEBUG

def Merge_2Borders(bord1, bord2):
    '''
    Merge selected 2 boundaries (if possible)
    '''
    inter =  np.intersect1d(bord1, bord2)    
    if len(inter) == 2:        
        ind1 = [np.flatnonzero(bord1 == i) for i in inter]
        ind2 = [np.flatnonzero(bord2 == i) for i in inter]
        res[0] = inter[0]
        act_ind = ind1[0]      
    else:   
        return [bord1, bord2]

#%%
import sympy as sp


a,b,c,a1,b1,c1,x0,y0,x1,y2 = sp.symbols('a,b,c,a1,b1,c1,x0,y0,x1,y2')
sp.solve([a*x0+b*y0+c, a1*x0+b1*y0+c1], x0, y0)



#%%
ind = 0
act_point = relay_const[ind]
cc = np.hstack([relay_const[:ind],relay_const[(ind+1):]]) - act_point
plt.plot(np.real(cc), np.imag(cc), 'ok', ms=10)
plt.plot(0,0, 'og', ms=10)

points = 0.5 * np.hstack([cc[:ind],cc[(ind+1):]]) # mid  points


plt.plot(np.real(points), np.imag(points), 'xg', ms=5)

#%% preparation of lines
plt.figure(figsize(12,12))
c1=[]
c2=[]
mp = []
lins = []
N = 0
plt.plot(0,0,'or', ms=15)
Points = np.hstack([-1.5,1.5,-1.5j,1.5j, np.exp(1j*np.random.rand(N)*2*np.pi)*(1+0.1*np.random.rand(N))])

for p in Points:
    px = np.real(p)+np.imag(p)**2/np.real(p) # cross x axis
    py = np.abs(p)/np.sin(np.angle(p))*1j # cross y axis
    c = np.imag(py)
    a = -c/np.real(px)
    b = -1
    if np.abs(np.imag(p)) < 1e-10:
        py = np.inf # no crossing y (imag) axis        
        c = px
        a = -1
        b = 0        
    if np.abs(np.real(p)) < 1e-10:
        px = np.inf # no crossing x (real) axis     
        a = 0  
        c = float(np.imag(py))
#    # check
#    c1.append(np.abs(px)**2 - np.abs(p-px)**2 - np.abs(p)**2)
#    c2.append(np.abs(p)**2+np.abs(py-p)**2-np.abs(py)**2)
    lins.append([a,b,c])
    plt.plot(np.real(p), np.imag(p), 'og', ms=5)
    ran = np.array([-2,2])
    if b == 0:
        #plt.plot([c, c],ran, '--k')
        mp.append(np.array([c, c])+ran*1j)
    else:
        #plt.plot(ran,a*ran+c, '--k')
        mp.append(ran+(a*ran+c)*1j)

lins = np.asarray(lins)
   
for i in mp:
    plt.plot(np.real(i), np.imag(i), '--k')
    

selected = []
sel_intersect = []
bord_lines = []

ind_sel = np.argmin(np.abs(Points)) # the first selected point (index)
selected.append(ind_sel)

while True:
    point_sel = Points[ind_sel] 
    lin_sel = lins[ind_sel]
    
    P_rem = Points * np.exp(-1j*np.angle(point_sel)) # rotation that selected point has angle zero    
    sat_inds, = np.nonzero((np.mod(np.angle(P_rem), 2*np.pi) > 1e-10)*(np.mod(np.angle(P_rem), 2*np.pi) < np.pi - 1e-10)) # positive indecis    
    intersects = intersect(lin_sel, lins[sat_inds])    
    print point_sel, intersects, sat_inds
    dists = np.abs(intersects - point_sel)
    ind_min = np.argmin(dists)
    sel_intersect.append(intersects[ind_min])
    ind_sel = sat_inds[ind_min]
    if ind_sel == selected[0]:
        break
    else:
        selected.append(ind_sel)       
borders = np.hstack([sel_intersect, sel_intersect[0]])


plt.plot(np.real(borders), np.imag(borders), 'k-', lw=2)
