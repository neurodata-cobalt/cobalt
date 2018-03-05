import numpy as np
import skfmm
from scipy.ndimage.filters import laplace

class vertices:
    def __init__(self, mask , vol):
        '''
        Constructor
        '''
        self.mask = np.copy(mask) # binarized volume
        self.masked_vol = np.copy(vol)
        self.masked_vol[self.mask == 0] = 0
        
        

    def compute_w1(self):
        '''
        Compute W1 for vertex weight computation
        '''
        min_intensity = np.float64(np.min(self.masked_vol[self.masked_vol !=0 ]))
        max_intensity = np.max(self.masked_vol)
        w1 = np.copy(self.masked_vol)
        w1 = (w1 - min_intensity)/(max_intensity - min_intensity)
        w1[w1 < 0] = 0

        return w1
    
    def compute_w2(self):
        '''
        Compute W2 for vertex weight computation; Distance from boundaries.
        '''
        w2 = np.copy(self.masked_vol)
        w2 = skfmm.distance(w2)
        min_distance = np.float64(np.min(w2[w2 != 0]))
        max_distance = np.max(w2)
        w2 = (w2 - min_distance)/(max_distance - min_distance)
        w2[w2 < 0] = 0
        
        return w2
    
    def compute_w3(self):
        '''
        Compute W3 for vertex weight computation; Laplacian of edges
        '''
        w2 = self.compute_w2()
        lplc = laplace(w2)
        lplc = np.tanh(2.0*lplc)
        w3 = np.minimum(np.ones(lplc.shape) , 1.0 - lplc)
        w3[self.mask == 0] = 0
        return w3
        
    def compute_vertex_wight(self):
        '''
        Compute vetex weight (average of w1, w2, and w3)
        '''
        w1 = self.compute_w1()
        w2 = self.compute_w2()
        w3 = self.compute_w3()
        
        vw = (w1 + w2 + w3)/3.0
#         vw[self.mask == 0] = 0
        
        return vw
            
