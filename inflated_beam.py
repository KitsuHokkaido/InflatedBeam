from dolfinx import log, default_scalar_type
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, mesh

class InflatedBeam:
    def __init__(self, L, R, h, nb_elts, degree, E=2e5, v=0.3):
        self.degree = degree
        self.nb_elts = nb_elts
        
        # Grandeurs du système (en cm)
        self.L = L
        self.R = R
        self.h = h

        # Grandeurs comportements
        self.E = E
        self.v = v

        self.A = E*h
        self.D_1 = E*h**3/12*(1 - v**2)
        self.D_2 = self.D_1
        self.D_3 = v*self.D_1
        self.D_4 = E*h**3/24*(1 + v**2)


        self.domain = mesh.create_interval(MPI.COMM_WORLD, nb_elts, [0, L], dtype=np.float32)
        self.tdim = self.domain.topology.dim
    
    def setup_function_space(self):
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
    
    def get_data(self, ):
        return self.domain, self.tdim

#element = ufl.MixedElement([
#    ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree),  # u1
#    ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree),  # u3  
#    ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree),  # gamma
#    ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree)   # alpha
#])
#V_mixed = dolfinx.fem.FunctionSpace(domain, element)

