from typing import *
import time
import scipy.sparse.linalg as spla
from scipy import sparse
import numpy as np


def build_cotan_laplacian( points: np.ndarray, tris: np.ndarray ):
    # My fix - delete degenerate edges - part 1
    del_idx = np.where(tris[:, 0] == tris[:, 1])[0]
    tris = np.delete(tris, del_idx, axis=0)
    del_idx = np.where(tris[:, 0] == tris[:, 2])[0]
    tris = np.delete(tris, del_idx, axis=0)
    del_idx = np.where(tris[:, 1] == tris[:, 2])[0]
    tris = np.delete(tris, del_idx, axis=0)
    # End of part 1
    a,b,c = (tris[:,0],tris[:,1],tris[:,2])
    A = np.take( points, a, axis=1 )
    B = np.take( points, b, axis=1 )
    C = np.take( points, c, axis=1 )

    eab,ebc,eca = (B-A, C-B, A-C)
    # # My fix - handle degenerate edges - part 1
    # eab_deg_idx = np.where(np.sum(eab == 0, axis=0) == 3)[0]
    # print(eab_deg_idx.shape)
    # ebc_deg_idx = np.where(np.sum(ebc == 0, axis=0) == 3)[0]
    # print(ebc_deg_idx.shape)
    # eca_deg_idx = np.where(np.sum(eca == 0, axis=0) == 3)[0]
    # print(eca_deg_idx.shape)
    # # End of part 1
    eab = eab/np.linalg.norm(eab,axis=0)[None,:]
    ebc = ebc/np.linalg.norm(ebc,axis=0)[None,:]
    eca = eca/np.linalg.norm(eca,axis=0)[None,:]    

    # My fix - handle numerical errors - part 1
    t = -np.sum(eca*eab,axis=0)
    t_i = np.where(t>=1)[0]
    t[t_i] = 0.999
    t_i = np.where(t<=-1)[0]
    t[t_i] = -0.999
    alpha = np.arccos(t)
    t = -np.sum(eab*ebc,axis=0)
    t_i = np.where(t>=1)[0]
    t[t_i] = 0.999
    t_i = np.where(t<=-1)[0]
    t[t_i] = -0.999
    beta = np.arccos(t)
    t = -np.sum(ebc*eca,axis=0)
    t_i = np.where(t>=1)[0]
    t[t_i] = 0.999
    t_i = np.where(t<=-1)[0]
    t[t_i] = -0.999
    gamma = np.arccos(t)
    # alpha = np.arccos( -np.sum(eca*eab,axis=0) )
    # beta  = np.arccos( -np.sum(eab*ebc,axis=0) )
    # gamma = np.arccos( -np.sum(ebc*eca,axis=0) )
    # End of part 1

    wab,wbc,wca = ( 1.0/np.tan(gamma), 1.0/np.tan(alpha), 1.0/np.tan(beta) )
    # # My fix - handle degenerate edges - part 2
    # wab[eab_deg_idx] = 0.001
    # wbc[ebc_deg_idx] = 0.001
    # wca[eca_deg_idx] = 0.001
    # # End of part 2
    rows = np.concatenate((   a,   b,   a,   b,   b,   c,   b,   c,   c,   a,   c,   a ), axis=0 )
    cols = np.concatenate((   a,   b,   b,   a,   b,   c,   c,   b,   c,   a,   a,   c ), axis=0 )
    vals = np.concatenate(( wab, wab,-wab,-wab, wbc, wbc,-wbc,-wbc, wca, wca,-wca, -wca), axis=0 )
    L = sparse.coo_matrix((vals,(rows,cols)),shape=(points.shape[1],points.shape[1]), dtype=float).tocsc()
    return L


def build_weights_and_adjacency( points: np.ndarray, tris: np.ndarray, L: Optional[sparse.csc_matrix]=None ):
    L = L if L is not None else build_cotan_laplacian( points, tris )
    n_pnts, n_nbrs = (points.shape[1], L.getnnz(axis=0).max()-1)
    nbrs = np.ones((n_pnts,n_nbrs),dtype=int)*np.arange(n_pnts,dtype=int)[:,None]
    wgts = np.zeros((n_pnts,n_nbrs),dtype=float)

    for idx,col in enumerate(L):
        msk = col.indices != idx
        indices = col.indices[msk]
        values  = col.data[msk]
        nbrs[idx,:len(indices)] = indices
        wgts[idx,:len(indices)] = -values

    return nbrs, wgts, L


class ARAP:
    def __init__( self, points: np.ndarray, tris: np.ndarray, anchors: List[int], anchor_weight: Optional[float]=10.0, L: Optional[sparse.csc_matrix]=None ):
        self._pnts    = points.copy()
        self._tris    = tris.copy()
        self._nbrs, self._wgts, self._L = build_weights_and_adjacency( self._pnts, self._tris, L )

        self._anchors = list(anchors)
        self._anc_wgt = anchor_weight
        E = sparse.dok_matrix((self.n_pnts,self.n_pnts),dtype=float)
        for i in anchors:
            E[i,i] = 1.0
        E = E.tocsc()
        self._solver = spla.factorized( ( self._L.T@self._L + self._anc_wgt*E.T@E).tocsc() )

    @property
    def n_pnts( self ):
        return self._pnts.shape[1]

    @property
    def n_dims( self ):
        return self._pnts.shape[0]

    def __call__( self, anchors: Dict[int,Tuple[float,float,float]], num_iters: Optional[int]=4 ):
        con_rhs = self._build_constraint_rhs(anchors)
        R = np.array([np.eye(self.n_dims) for _ in range(self.n_pnts)])
        def_points = self._solver( self._L.T@self._build_rhs(R) + self._anc_wgt*con_rhs )
        for i in range(num_iters):
            R = self._estimate_rotations( def_points.T )
            def_points = self._solver( self._L.T@self._build_rhs(R) + self._anc_wgt*con_rhs )
            print("INFO: Done iteration {}/{}".format(i+1, num_iters))
        return def_points.T

    def _estimate_rotations( self, def_pnts: np.ndarray ):
        tru_hood = (np.take( self._pnts, self._nbrs, axis=1 ).transpose((1,0,2)) - self._pnts.T[...,None])*self._wgts[:,None,:]
        rot_hood = (np.take( def_pnts,   self._nbrs, axis=1 ).transpose((1,0,2)) - def_pnts.T[...,None])

        U,s,Vt = np.linalg.svd( rot_hood@tru_hood.transpose((0,2,1)) )
        R = U@Vt
        dets = np.linalg.det(R)
        Vt[:,self.n_dims-1,:] *= dets[:,None]
        R = U@Vt
        return R

    def _build_rhs( self, rotations: np.ndarray ):
        R = (np.take( rotations, self._nbrs, axis=0 )+rotations[:,None])*0.5
        tru_hood = (self._pnts.T[...,None]-np.take( self._pnts, self._nbrs, axis=1 ).transpose((1,0,2)))*self._wgts[:,None,:]
        rhs = np.sum( (R@tru_hood.transpose((0,2,1))[...,None]).squeeze(), axis=1 )
        return rhs

    def _build_constraint_rhs( self, anchors: Dict[int,Tuple[float,float,float]] ):
        f = np.zeros((self.n_pnts,self.n_dims),dtype=float)
        f[self._anchors,:] = np.take( self._pnts, self._anchors, axis=1 ).T
        for i,v in anchors.items():
            if i not in self._anchors:
                raise ValueError('Supplied anchor was not included in list provided at construction!')
            f[i,:] = v
        return f


def grid_mesh_2d( nx, ny, h ):
    x,y = np.meshgrid( np.linspace(0.0,(nx-1)*h,nx), np.linspace(0.0,(ny-1)*h,ny))
    idx = np.arange(nx*ny,dtype=int).reshape((ny,nx))
    quads = np.column_stack(( idx[:-1,:-1].flat, idx[1:,:-1].flat, idx[1:,1:].flat, idx[:-1,1:].flat ))
    tris  = np.vstack((quads[:,(0,1,2)],quads[:,(0,2,3)]))
    return np.row_stack((x.flat,y.flat)), tris, idx


def run():
    nx,ny,h = (21,5,0.1)
    pnts, tris, ix = grid_mesh_2d( nx,ny,h )

    anchors = {}
    d = 1
    mid = int(nx / 2)
    for i in range(ny):
        anchors[ix[i, 0]]    = (0.0, i*h)
        anchors[ix[i, 1]]    = (h, i*h)
        anchors[ix[i, nx-2]] = (h*nx-h, d+i*h)
        anchors[ix[i, nx-1]] = (h*nx, d+i*h)

        # anchors[ix[i, mid]] = (h*mid, (d+i*h) * 0.5 + 0.1)
        # anchors[ix[i, mid]] = (h*mid, (d+i*h) * 0.5 + 0.1)

    deformer = ARAP( pnts, tris, anchors.keys(), anchor_weight=20)
    start = time.time()
    def_pnts = deformer(anchors, num_iters=10)
    end = time.time()

    import matplotlib.pyplot as plt
    plt.triplot( pnts[0], pnts[1], tris, 'k-', label='original' )
    plt.triplot( def_pnts[0], def_pnts[1], tris, 'r-', label='deformed' )
    plt.legend()
    plt.title('deformed in {:0.2f}ms'.format((end-start)*1000.0))
    plt.show()

if __name__ == '__main__':
    run()
    exit()

