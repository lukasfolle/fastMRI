# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:58:20 2017
@author: sifeluga (Felix Lugauer)
"""

import argparse
import math
from os import getcwd

# for plotting the k-space/density
import matplotlib.pyplot as plt
import numpy as np


class PoissonSampler:
    # those are never changed
    f2PI = math.pi * 2.
    # minimal radius (packing constant) for a fully sampled k-space
    fMinDist = 0.634

    def __init__(self, M=110, N=50, AF=12.0, fPow=2.0, NN=47, aspect=0, tInc=0.):
        self.M = M
        self.N = N
        self.fAF = AF
        self.fAspect = aspect
        if aspect == 0:
            self.fAspect = M / N
        self.NN = NN
        self.fPow = fPow
        self.tempInc = tInc

        self.M2 = round(M / 2)
        self.N2 = round(N / 2)
        # need to store density matrix
        self.density = np.zeros((M, N), dtype=np.float32)
        self.targetPoints = round(M * N / AF)

        # init varDens
        if self.fPow > 0:
            self.variableDensity()

    def variableDensity(self):
        """Precomputes a density matrix, which is used to scale the location-dependent
        radius used for generating new samples.
         """
        fNorm = 1.2 * math.sqrt(pow(self.M2, 2.) + pow(self.N2 * self.fAspect, 2.))

        # computes the euclidean distance for each potential sample location to the center
        for j in range(-self.N2, self.N2, 1):
            for i in range(-self.M2, self.M2, 1):
                self.density[i + self.M2, j + self.N2] = (
                        1. - math.sqrt(math.pow(j * self.fAspect, 2.) + math.pow(i, 2.)) / fNorm)

        # avoid diving by zeros
        self.density[(self.density < 0.001)] = 0.001
        # raise scaled distance to the specified power (usually quadratic)
        self.density = np.power(self.density, self.fPow)
        accuDensity = math.floor(np.sum(self.density))

        # linearly adjust accumulated density to match desired number of samples
        if accuDensity != self.targetPoints:
            scale = self.targetPoints / accuDensity
            scale *= 1.0
            self.density *= scale
            self.density[(self.density < 0.001)] = 0.001

    def addPoint(self, ptN, fDens, iReg):
        """Inserts a point in the sampling mask if that point is not yet sampled
        and suffices a location-depdent distance (variable density) to
        neighboring points. Returns the index > -1 on success."""
        ptNew = np.around(ptN).astype(int, copy=False)
        idx = ptNew[0] + ptNew[1] * self.M

        # point already taken
        if self.mask[ptNew[0], ptNew[1]]:
            return -1

        # check for points in close neighborhood
        for j in range(max(0, ptNew[1] - iReg), min(ptNew[1] + iReg, self.N), 1):
            for i in range(max(0, ptNew[0] - iReg), min(ptNew[0] + iReg, self.M), 1):
                if self.mask[i, j] == True:
                    pt = self.pointArr[self.idx2arr[i + j * self.M]]
                    if pow(pt[0] - ptN[0], 2.) + pow(pt[1] - ptN[1], 2.) < fDens:
                        return -1

        # success if no point was too close
        return idx

    def generate(self, rng, accu_mask=None):

        # set seed for deterministic results
        self.rng_new = rng

        # preset storage variables
        self.idx2arr = np.zeros((self.M * self.N), dtype=np.int32)
        self.idx2arr.fill(-1)
        self.mask = np.zeros((self.M, self.N), dtype=bool)
        self.mask.fill(False)
        self.pointArr = np.zeros((self.M * self.N, 2), dtype=np.float32)
        activeList = []

        # inits
        count = 0
        pt = np.array([self.M2, self.N2], dtype=np.float32)

        # random jitter of inital point
        jitter = 4
        pt += self.rng_new.uniform(-jitter / 2, jitter / 2, 2)

        # update: point matrix, mask, current idx, idx2matrix and activeList
        self.pointArr[count] = pt
        ptR = np.around(pt).astype(int, copy=False)
        idx = ptR[0] + ptR[1] * self.M
        self.mask[ptR[0], ptR[1]] = True
        self.idx2arr[idx] = count
        activeList.append(idx)
        count += 1

        # uniform density
        if self.fPow == 0:
            self.fMinDist *= self.fAF

        # now sample points
        while activeList:
            idxp = activeList.pop()
            curPt = self.pointArr[self.idx2arr[idxp]]
            curPtR = np.around(curPt).astype(int, copy=False)

            fCurDens = self.fMinDist
            if (self.fPow > 0):
                fCurDens /= self.density[curPtR[0], curPtR[1]]

            region = int(round(fCurDens))

            # if count >= self.targetPoints:
            #    break

            # try to generate NN points around an arbitrary existing point
            for i in range(0, self.NN):
                # random radius and angle
                fRad = self.rng_new.uniform(fCurDens, fCurDens * 2.)
                fAng = self.rng_new.uniform(0., self.f2PI)

                # generate new position
                ptNew = np.array([curPt[0], curPt[1]], dtype=np.float32)
                ptNew[0] += fRad * math.cos(fAng)
                ptNew[1] += fRad * math.sin(fAng)
                ptNewR = np.around(ptNew).astype(int, copy=False)
                # continue when old and new positions are the same after rounding
                if ptNewR[0] == curPtR[0] and ptNewR[1] == curPtR[1]:
                    continue

                if ptNewR[0] >= 0 and ptNewR[1] >= 0 and ptNewR[0] < self.M and ptNewR[1] < self.N:
                    newCurDens = self.fMinDist / self.density[ptNewR[0], ptNewR[1]]
                    if self.fPow == 0:
                        newCurDens = self.fMinDist
                    if self.tempInc > 0 and accu_mask is not None:
                        if accu_mask[ptNewR[0], ptNewR[1]] >\
                                self.density[ptNewR[0], ptNewR[1]] + 1.01 - self.tempInc:
                            continue
                    idx = self.addPoint(ptNew, newCurDens, region)
                    if idx >= 0:
                        self.mask[ptNewR[0], ptNewR[1]] = True
                        self.pointArr[count] = ptNew
                        self.idx2arr[idx] = count
                        activeList.append(idx)
                        count += 1
        return self.mask


if __name__ == '__main__':
    from copy import deepcopy
    from time import time
    rng = np.random.default_rng()

    before = time()
    accu_inner = np.zeros((10, 64), dtype=np.int32)
    PS = PoissonSampler(10, 64, 4, 2, 42, 0, 0.7)
    mask_inner = PS.generate(rng, accu_inner)
    accu_outer = np.zeros((20, 128), dtype=np.int32)
    PS = PoissonSampler(20, 128, 9, 2, 42, 0, 0.7)
    mask_outer = PS.generate(rng, accu_outer)
    mask_combined = deepcopy(mask_outer)
    mask_combined[5:-5, 32:-32] = mask_inner
    print(f"Accel: {np.prod(mask_combined.shape) / np.sum(mask_combined)}")
    after = time()
    print(f"Time passed {after - before}sec")
    plt.subplot(1, 3, 1)
    plt.imshow(mask_inner)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_outer)
    plt.subplot(1, 3, 3)
    plt.imshow(mask_combined)
    plt.show()
