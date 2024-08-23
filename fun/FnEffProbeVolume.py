import numpy as np
def FnEffProbeVolume(p1, p2, p3):
# FnEffProbeVolume(p1, p2, p3, r1, r2, r3) a function to calculate the effective probe volume with simplified assumptions
#
# Method:
# The beam intersects at a point with different beam probe volume. this results in an intersection volume
# which is deducted from the total volume of three spheres

# Syntax:  

# Inputs:
    # p1, p2, p3 - position array, array of 3 elements e.g. p1 = (x1, y1, z1)
    # r1, r2, r3 - radis array for 3 beams e.g. r1 = 10m at 150m focus
    # var - variable to be plotted (array of 3) (var0, var1, var2)
    # label - label for [x, y, var[0], var[1], var[2], colorbar]
    # filename - path and file.extension for the saving plot (see matplotlib savefig formats)
#
# Outputs:
#    dZ_eff - effective probe volume with intersecting spheres
#    dZ - theoretical probe volume for CW Lidars
#    dL - sum of separation distance between the points
#    Leff  - effective probe volume at distance L for 3 intersecting spheres
 
#
# Example:

# modules required: numpy, matplotlib
# classes required: none
# Data-files required: none
#
# See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2

# References:
#
# Author: Ashim Giyanani, Research Associate
# Fraunhofer Institute of Wind Energy
# Windpark planning and operation department
# Am Seedeich 45, Bremerhaven
# email: ashim.giyanani@iwes.fraunhofer.de
# Git site: https://gitlab.cc-asp.fraunhofer.de/giyash/testfeld-bhv.git
# Created: 06-08-2020; Last revision: 14.01.2022 11:42:08

    #------------- BEGIN CODE --------------
    #
    import sys
    sys.path.append(r'c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules')
    from FnLidarFwhmCW import FnLidarFwhmCW
    from threespheres import triple_overlap

    # Input data from DTU
    lambda0 = 1.55e-6 # laser wavelength
    w0 = [32e-3, 32e-3, 34e-3] # aperture radius

    # unpack positions
    x1, y1, z1 = p1[0], p1[1], p1[2]
    x2, y2, z2 = p2[0], p2[1], p2[2]
    x3, y3, z3 = p3[0], p3[1], p3[2]

    # centroid of the three points assuming triangular constellation
    xc = np.nanmean([x1, x2, x3], axis=0, where=~(np.isnan(x1)|np.isnan(x2)|np.isnan(x3)))
    yc = np.nanmean([y1, y2, y3], axis=0, where=~(np.isnan(y1)|np.isnan(y2)|np.isnan(y3)))
    zc = np.nanmean([z1, z2, z3], axis=0, where=~(np.isnan(z1)|np.isnan(z2)|np.isnan(z3)))

    # find the distance between the centroid and vertices of the triangle
    c1 = np.sqrt((np.nansum([x1,-xc], axis=0))**2 + (np.nansum([y1,-yc], axis=0))**2 + (np.nansum([z1,-zc], axis=0))**2)
    c2 = np.sqrt((np.nansum([x2,-xc], axis=0))**2 + (np.nansum([y2,-yc], axis=0))**2 + (np.nansum([z2,-zc], axis=0))**2)
    c3 = np.sqrt((np.nansum([x3,-xc], axis=0))**2 + (np.nansum([y3,-yc], axis=0))**2 + (np.nansum([z3,-zc], axis=0))**2)
    cmax = np.nanmax((c1, c2, c3), axis=0, initial=0, where=~np.isnan(c1))
    # the effective probe length in the
    Leff = 2*cmax

    # calculate the focus distance from p1, p2, p3
    L1 = np.sqrt(x1**2 + y1**2 + z1**2)
    L2 = np.sqrt(x2**2 + y2**2 + z2**2)
    L3 = np.sqrt(x3**2 + y3**2 + z3**2)

    # sum of separation, assuming orthogonal projections
    dL12 = np.sqrt((np.nansum([x1,-x2], axis=0))**2 + (np.nansum([y1,-y2], axis=0))**2 + (np.nansum([z1,-z2], axis=0))**2)
    dL23 = np.sqrt((np.nansum([x2,-x3], axis=0))**2 + (np.nansum([y2,-y3], axis=0))**2 + (np.nansum([z2,-z3], axis=0))**2)
    dL31 = np.sqrt((np.nansum([x3,-x1], axis=0))**2 + (np.nansum([y3,-y1], axis=0))**2 + (np.nansum([z3,-z1], axis=0))**2)
    dL = dL12 + dL23 + dL31

    # calculate the probe volume at the focus distance
    dZ1, dZ2, dZ3 = np.zeros(len(x1)), np.zeros(len(x1)), np.zeros(len(x1))
    vol_overlap = np.zeros(len(x1))
    r1, r2, r3 = np.zeros(len(x1)), np.zeros(len(x1)), np.zeros(len(x1))
    idx = []
    for i in np.arange((len(L1))):
        dZ1[i],_ = FnLidarFwhmCW(lambda0, w0[0], L1[i], plotFig=0)
        dZ2[i],_ = FnLidarFwhmCW(lambda0, w0[1], L2[i], plotFig=0)
        dZ3[i],_ = FnLidarFwhmCW(lambda0, w0[2], L3[i], plotFig=0)

        # beam radius i.e. half of beam width
        r1[i], r2[i], r3[i] = dZ1[i]/2, dZ2[i]/2, dZ3[i]/3
        # intersection volume of three spheres
        # vol_overlap[i] = triple_overlap(p1[:,i], p2[:,i], p3[:,i], r1[i], r2[i], r3[i], mc_check=False)
        vol_overlap[i]=0
        # if there is no intersection, take dL as dZ_eff
        if vol_overlap[i] <= 0:
            # print('no intersection at the actual point')
            idx.append(i)

    # volume of spheres at focus point
    vol1, vol2, vol3 = (4*np.pi)*(r1**3)/3, (4*np.pi)*(r2**3)/3, (4*np.pi)*(r3**3)/3
    # find the effective total volume of the spheres - intersecction volume
    vol_eff = vol1 + vol2 + vol3 - vol_overlap
     
    # assuming effective volume to be spherical, find the diameter of the sphere
    r_eff = (3*vol_eff/(4*np.pi))**(1/3)
    # effective probe volume at distance L
    dZ_eff = 2*r_eff
    # effective probe volume at distance L for no 3 intersecting spheres
    dZ_eff[idx] = dL[idx]/3 + 2*(r1[idx] + r2[idx] + r3[idx])/3

    dZ = np.array((dZ1, dZ2, dZ3))

    return dZ_eff, dZ, dL, Leff

# Example:
if __name__ == "__main__":
    p1 = np.array([[-0.4923304544238931, -15.880610942283525, 124.52975729103373], [-0.4923304544238931, -15.880610942283525, 124.52975729103373]])
    p2 = np.array([[0.46010120269767185, -15.239468177359228, 124.49987435365591],[0.46010120269767185, -15.239468177359228, 124.49987435365591]])
    p3 = np.array([[1.7162217581114305, -14.643173882739973, 125.53931388776589],[1.7162217581114305, -14.643173882739973, 125.53931388776589]])
    p1[0] = np.nan
    dZ_eff, dZ, dL, Leff = FnEffProbeVolume(p1.T, p2.T, p3.T)

    # theoretical probe length
    r1 = np.linspace(0, 300)
    phi = np.deg2rad(30)
    x1 = r1 * np.cos(phi)
    y1 = np.zeros(len(r1))
    z1 = r1 * np.sin(phi)
    p1 = np.array((x1, y1, z1))
    p2 = p1.copy()
    p3 = p1.copy()
    dZ_eff, dZ, dL, Leff = FnEffProbeVolume(p1.T, p2.T, p3.T)



