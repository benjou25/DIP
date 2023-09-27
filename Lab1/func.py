import numpy as np

#Stretches an image in hight, doubles the size ("fac" times)
def stretch(fac, xtemp, ytemp):
    for j in range(fac):
        rescalex = np.zeros((2*np.size(xtemp[:,0]), np.size(xtemp[0,:])))
        rescaley = np.zeros((2*np.size(ytemp[:,0]), np.size(ytemp[0,:])))
        for k in range(np.size(xtemp[:,0])):
            rescalex[2*k, :] = xtemp[k, :]
            rescaley[2*k, :] = ytemp[k, :]
            if k < np.size(xtemp[:,0])-1:
                rescalex[2*k+1, :] = (xtemp[k, :] + xtemp[k+1, :])/2
                rescaley[2*k+1, :] = (ytemp[k, :] + ytemp[k+1, :])/2
        xtemp = rescalex
        ytemp = rescaley
    return rescalex, rescaley