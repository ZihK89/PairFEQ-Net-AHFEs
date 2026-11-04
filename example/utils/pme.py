"""PME charge derivatives, including real, reciprocal, self and exclusion terms.

Order-five splines follow OpenMM 8.2 ReferencePME.cpp:
https://github.com/openmm/openmm/blob/8.2.0/platforms/reference/src/SimTKReference/ReferencePME.cpp
Fixed exception charge products do not depend on the predicted charges.
"""
import math

import numpy as np
from numba import njit, prange
from scipy.fft import fftn, rfftn, irfftn
from openmm import unit

COULOMB = 1.602176634e-19**2 * 6.02214076e23 / (4*np.pi*8.8541878128e-12*1e-6)
ORDER = 5


@njit(cache=True)
def _spline_weights(fractions):
    """Cardinal B-spline weights, spreading forward from the containing cell."""
    n = fractions.shape[0]
    weights = np.zeros((n, 3, ORDER), dtype=np.float64)
    for atom in range(n):
        for axis in range(3):
            f = fractions[atom, axis]
            weights[atom, axis, 0] = 1-f
            weights[atom, axis, 1] = f
            for degree in range(2, ORDER):
                weights[atom, axis, degree] = f*weights[atom, axis, degree-1]/degree
                for index in range(degree-1, 0, -1):
                    weights[atom, axis, index] = (
                        (f+degree-index)*weights[atom, axis, index-1]
                        +(index+1-f)*weights[atom, axis, index])/degree
                weights[atom, axis, 0] *= (1-f)/degree
    return weights


@njit(cache=True)
def _interpolate(mesh, indices, weights, selected):
    values = np.zeros(selected.shape[0], dtype=np.float64)
    for i in range(selected.shape[0]):
        atom = selected[i]
        for ix in range(ORDER):
            gx = (indices[atom, 0]+ix) % mesh.shape[0]
            for iy in range(ORDER):
                gy = (indices[atom, 1]+iy) % mesh.shape[1]
                wxy = weights[atom, 0, ix]*weights[atom, 1, iy]
                for iz in range(ORDER):
                    gz = (indices[atom, 2]+iz) % mesh.shape[2]
                    values[i] += wxy*weights[atom, 2, iz]*mesh[gx, gy, gz]
    return values


def _moduli(size):
    weights = _spline_weights(np.zeros((1, 3)))[0, 0]
    data = np.zeros(size)
    data[1:ORDER+1] = weights
    moduli = np.abs(fftn(data))**2
    for i in range(size):
        if moduli[i] < 1e-7:
            moduli[i] = 0.5*(moduli[(i-1) % size]+moduli[(i+1) % size])
    return moduli


@njit(cache=True)
def spread_wrapped(indices, weights, charges, shape):
    """Preserve deposition order; precompute periodic stencil indices."""
    mesh = np.zeros((shape[0],shape[1],shape[2]), dtype=np.float64)
    wrapped = np.empty((indices.shape[0],3,5),dtype=np.int64)
    for atom in range(indices.shape[0]):
        for axis in range(3):
            for k in range(5):
                wrapped[atom,axis,k]=(indices[atom,axis]+k)%shape[axis]
    for atom in range(charges.shape[0]):
        for ix in range(5):
            gx=wrapped[atom,0,ix]
            qx=charges[atom]*weights[atom,0,ix]
            for iy in range(5):
                gy=wrapped[atom,1,iy]
                qxy=qx*weights[atom,1,iy]
                for iz in range(5):
                    gz=wrapped[atom,2,iz]
                    mesh[gx,gy,gz]+=qxy*weights[atom,2,iz]
    return mesh


@njit(cache=True, inline='always')
def one_potential(i, positions, charges, box, selected, excluded,
                  alpha, cutoff, exceptions_periodic):
    atom = selected[i]
    total = 0.0
    cutoff2 = cutoff*cutoff
    for other in range(charges.shape[0]):
        if other == atom:
            continue
        dx = positions[other, 0] - positions[atom, 0]
        dy = positions[other, 1] - positions[atom, 1]
        dz = positions[other, 2] - positions[atom, 2]
        exception = excluded[i, other]
        if not exception or exceptions_periodic:
            shift = math.floor(dz/box[2, 2] + 0.5)
            dx -= shift*box[2, 0]
            dy -= shift*box[2, 1]
            dz -= shift*box[2, 2]
            shift = math.floor(dy/box[1, 1] + 0.5)
            dx -= shift*box[1, 0]
            dy -= shift*box[1, 1]
            shift = math.floor(dx/box[0, 0] + 0.5)
            dx -= shift*box[0, 0]
        r2 = dx*dx + dy*dy + dz*dz
        if exception:
            if r2 == 0:
                total -= charges[other]*2*alpha/math.sqrt(math.pi)
            else:
                r = math.sqrt(r2)
                total -= charges[other]*math.erf(alpha*r)/r
        elif r2 < cutoff2:
            r = math.sqrt(r2)
            total += charges[other]*math.erfc(alpha*r)/r
    total -= 2*alpha/math.sqrt(math.pi)*charges[atom]
    return COULOMB*total


@njit(cache=True, parallel=True)
def direct_parallel(positions, charges, box, selected, excluded,
                    alpha, cutoff, exceptions_periodic):
    result = np.zeros(selected.shape[0], dtype=np.float64)
    for i in prange(selected.shape[0]):
        result[i] = one_potential(i, positions, charges, box, selected, excluded,
                                  alpha, cutoff, exceptions_periodic)
    return result


class PMEPotential:
    """One PME mesh solve for dU/dq at the caffeine atoms."""

    def __init__(self, nonbonded, context, n_solute):
        # CPU PME initializes its actual FFT dimensions lazily.
        context.getState(getEnergy=True)
        alpha, nx, ny, nz = nonbonded.getPMEParametersInContext(context)
        self.alpha = float(alpha)
        self.shape = (int(nx), int(ny), int(nz))
        self.cutoff = nonbonded.getCutoffDistance().value_in_unit(unit.nanometer)
        self.exceptions_periodic = nonbonded.getExceptionsUsePeriodicBoundaryConditions()
        self.indices = np.arange(n_solute, dtype=np.int64)
        self.excluded = np.zeros((n_solute, nonbonded.getNumParticles()), dtype=np.bool_)
        for index in range(nonbonded.getNumExceptions()):
            first, second, *_ = nonbonded.getExceptionParameters(index)
            if first < n_solute:
                self.excluded[first, second] = True
            if second < n_solute:
                self.excluded[second, first] = True
        self.moduli = [_moduli(size) for size in self.shape]
        self.box = None

    def _setup_box(self, box):
        if np.any(np.abs(np.triu(box, 1)) > 1e-12) or np.any(np.diag(box) < 2*self.cutoff):
            raise ValueError("Expected a reduced OpenMM box with diagonals >= twice the cutoff")
        self.box = box.copy()
        self.inverse_box = np.linalg.inv(box)
        frequencies = [np.fft.fftfreq(size)*size for size in self.shape]
        wavevectors = np.stack(np.meshgrid(*frequencies, indexing="ij"), axis=-1)
        k2 = np.sum((wavevectors @ self.inverse_box.T)**2, axis=-1)
        b = self.moduli[0][:, None, None]*self.moduli[1][None, :, None]*self.moduli[2][None, None, :]
        k2[0, 0, 0] = 1.0
        influence = COULOMB*np.exp(-(np.pi/self.alpha)**2*k2)/(np.pi*np.linalg.det(box)*k2*b)
        influence[0, 0, 0] = 0.0
        # Symmetrization preserves real(ifftn(fftn(mesh)*H)) on triclinic Nyquist planes.
        negative = np.ix_(*[(-np.arange(n)) % n for n in self.shape])
        symmetric = 0.5*(influence + influence[negative])
        self.influence = np.ascontiguousarray(symmetric[:, :, :self.shape[2]//2+1])

    def potentials(self, positions, charges, box):
        if self.box is None or not np.array_equal(self.box, box):
            self._setup_box(box)
        fractional = positions @ self.inverse_box
        grid_positions = (fractional - np.floor(fractional))*np.asarray(self.shape)
        indices = np.floor(grid_positions).astype(np.int64)
        weights = _spline_weights(grid_positions - indices)
        mesh = spread_wrapped(indices, weights, charges, self.shape)
        potential = irfftn(rfftn(mesh)*self.influence, s=self.shape)*np.prod(self.shape)
        reciprocal = _interpolate(potential, indices, weights, self.indices)
        direct = direct_parallel(positions, charges, box, self.indices, self.excluded,
                                 self.alpha, self.cutoff, self.exceptions_periodic)
        phi = reciprocal + direct
        return phi - phi.mean()
