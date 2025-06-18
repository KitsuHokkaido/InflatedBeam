import numpy as np
import ufl

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type

from typing import Callable, Optional, Union

class InflatedBeam:
    def __init__(self, L:float, R:float, h:float, nb_elts:int, degree:int, E=2e5, v=0.3) -> None:
        """
        Initialise le modèle de poutre.
        
        Args:
            L: Longueur de la poutre (cm)
            R: Rayon initial (cm)  
            h: Épaisseur (cm)
            nb_elts: Nombre d'éléments finis
            degree: Degré des éléments
        """

        self._degree = degree
        self._nb_elts = nb_elts
        
        # Grandeurs du systeme (en cm)
        self._L = L
        self._R = R
        self._h = h

        # Grandeurs comportements
        self._E = E
        self._v = v

        self._A = E*h
        self._D1 = E*h**3/12*(1 - v**2)
        self._D2 = self._D1
        self._D3 = v*self._D1
        self._D4 = E*h**3/24*(1 + v**2)

        # Efforts extérieurs
        self._p = None
        self._f1 = None
        self._f3 = None
        self._c_gamma = None
        
        # Efforts extérieurs en un pt 
        self._point_moment = []
        self._point_forces = []

        self._pt_moment_ref = []
        self._pt_forces_ref = []

        self._domain = mesh.create_interval(MPI.COMM_WORLD, nb_elts, [0.0, L])
        self._tdim = self._domain.topology.dim

        self._setup_function_space()
    
    def _setup_function_space(self):
        self._V = fem.functionspace(self._domain, ("Lagrange", self._degree, (5,)))

        self._v_test = ufl.TestFunction(self._V)   
        self._v1, self._v3, self._v_gamma, self._v_alpha, self._v_lam = ufl.split(self._v_test)
        
        self._u_sol = fem.Function(self._V)
        self._u1_sol, self._u3_sol, self._gamma_sol, self._alpha_sol, self._lam_sol = ufl.split(self._u_sol)
    
    def _setup_variational_form(self):
        u1_1 = self._u1_sol.dx(0)
        u3_1 = self._u3_sol.dx(0)
        alpha_1 = self._alpha_sol.dx(0)
        alpha_11 = alpha_1.dx(0)

        e_p = u1_1 + 0.5 * (u1_1**2 + u3_1**2) 
        k_p = self._gamma_sol.dx(0)
        k_22s = self._R*(self._alpha_sol*ufl.sqrt(2 - self._alpha_sol**2) - 1)

        H = ( 2 * alpha_1**2 + alpha_11*(2*self._alpha_sol + self._alpha_sol**3) ) / (2 - self._alpha_sol**2)**(3/2)

        overline_es_squared = (self._R**2 * alpha_1**2 * ufl.pi ) / (2 - self._alpha_sol**2)

        overline_k11s_squared = (self._R**2*ufl.pi/4) * (3*(2 - self._alpha_sol**2)*(alpha_11)**2 - 2*ufl.sqrt(2 - self._alpha_sol**2)*alpha_11*H + 3*self._alpha_sol**2*H**2)

        overline_k21s_squared = (self._R**2 * alpha_1**2 * ufl.pi) / (2 - self._alpha_sol**2)

        overline_k11s = self._R * ufl.pi * (self._alpha_sol*H - alpha_11*ufl.sqrt(2 - self._alpha_sol**2))

        U_p = (self._R/2) * (
            2*ufl.pi * self._A * e_p**2 + 
            (self._A * self._R**2 * (2 - self._alpha_sol**2) * ufl.pi + 
             self._D1 * self._alpha_sol**2 * ufl.pi) * k_p**2
        )
        
        U_s = (self._R/2) * (
            self._A * overline_es_squared + 
            self._D1 * overline_k11s_squared + 
            self._D2 * k_22s**2 + 
            4*self._D4 * overline_k21s_squared + 
            2*self._D3 * k_22s * overline_k11s
        )
        
        U_ps = self._R * self._A * ufl.pi * e_p * self._R**2 * (alpha_1**2) / (2 - self._alpha_sol**2)
        
        energy_density = U_p + U_s + U_ps
        self._total_internal_energy = energy_density * ufl.dx
        
        self._setup_lagrange_multiplier()
        
        self._total_potential = self._total_internal_energy - self._total_external_work + self._lagrange_term 
        
        self._F = ufl.derivative(self._total_potential, self._u_sol, self._v_test)

    def _setup_lagrange_multiplier(self):
        u1_1 = self._u1_sol.dx(0)
        u3_1 = self._u3_sol.dx(0)
        j_p = ufl.sqrt((1 + u1_1)**2 + u3_1**2)

        C = - ufl.sin(self._gamma_sol)*(1 + u1_1)/j_p - ufl.cos(self._gamma_sol)*u3_1/j_p
    
        constraint_term_density = self._lam_sol * C
        self._lagrange_term = constraint_term_density * ufl.dx

    def _setup_external_work(self):

        x = ufl.SpatialCoordinate(self._domain)[0]
        
        if self._p is None:
            p = 0.0
        else:
            p = self._p(x)

        if self._f1 is None:
            f1 = 0
        else:
            f1 = self._f1(x)

        if self._f3 is None:
            f3 = 0
        else:
            f3 = self._f3(x)

        if self._c_gamma is None:
            c_gamma = 0
        else:
            c_gamma = self._c_gamma(x)

        
        pressure_work = p*ufl.pi*self._R**2*(self._alpha_sol*ufl.cos(self._gamma_sol) + ufl.sqrt(2 - self._alpha_sol**2)*(1 - 2*self._alpha_sol))
        force_work = f1*self._u1_sol + f3*self._u3_sol
        moment_work = c_gamma*self._gamma_sol

        point_moment_work = 0
        point_force_work = 0

        for pos, moment in self._point_moment:
            delta = self._create_point_source(pos)
            point_moment_work += moment * self._gamma_sol * delta

        for pos, f1, f3 in self._point_forces:
            delta = self._create_point_source(pos)
            point_force_work += f1 * self._u1_sol * delta + f3 * self._u3_sol * delta

        external_work_density = pressure_work + force_work + moment_work + point_moment_work + point_force_work
        self._total_external_work = external_work_density * ufl.dx


    def _apply_boundary_conditions(self):
        def boundary_left(x):
            return np.isclose(x[0], 0.0)

        def boundary_right(x):
            return np.isclose(x[0], self._L)

        boundary_facets_left = mesh.locate_entities_boundary(self._domain, self._tdim-1, boundary_left)
        boundary_facets_right = mesh.locate_entities_boundary(self._domain, self._tdim-1, boundary_right)
        
        dofs_left = [fem.locate_dofs_topological(self._V.sub(i), self._tdim-1, boundary_facets_left) for i in range(4)]
        bc_left_values = [0.0, 0.0, 0.0, 1.0]
        bc_left = [fem.dirichletbc(fem.Constant(self._domain, default_scalar_type(bc_left_values[i])), dofs_left[i], self._V.sub(i)) for i in range(4)] 
        
        dofs_right = [fem.locate_dofs_topological(self._V.sub(i), self._tdim-1, boundary_facets_right) for i in range(4)]
        bc_right_values = [0.0, 0.0, 0.0, 1.0]
        bc_right = [fem.dirichletbc(fem.Constant(self._domain, default_scalar_type(bc_right_values[i])), dofs_right[i], self._V.sub(i)) for i in range(4)]
        
        
        self._bcs = [bc_left[0], bc_left[1], bc_left[3], bc_right[1], bc_right[3]]

    def _set_initial_geometry(self):
        # Initialisation de la solution avec des valeurs physiques
        with self._u_sol.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        
        # On récup ici l'ensemble des positions des champs alpah et gamma au niveau des noeuds 
        alpha_dofs = self._V.sub(3).dofmap.list
        gamma_dofs = self._V.sub(2).dofmap.list
        
        # et ici on applique donc à tous les noeuds
        self._u_sol.x.array[alpha_dofs] = 1.0 
        self._u_sol.x.array[gamma_dofs] = 0.0


    def _apply_load_factor(self, factor, original_p, original_c_gamma, original_f1, original_f3):    
        # Appliquer le facteur de charge
        if self._p is not None:
            self._p = lambda x: factor * original_p(x)
        if self._c_gamma is not None:
            self._c_gamma = lambda x: factor * original_c_gamma(x)

        if self._f1 is not None:
            self._f1 = lambda x: factor * original_f1(x)
        if self._f3 is not None:
            self._f3 = lambda x: factor * original_f3(x)
       
        # Appliquer le facteur de charge aux points sources
        self._point_moment = [(pos, moment * factor) for pos, moment in self._pt_moment_ref]
    
        self._point_forces = [(pos, f1 * factor, f3 * factor) for pos, f1, f3 in self._pt_forces_ref]


    def _solve_step(self, step_index): 
        problem = NonlinearProblem(self._F, self._u_sol, self._bcs)
        solver = NewtonSolver(self._domain.comm, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6
        solver.atol = 1e-8
        solver.max_it = 50
        
        ksp = solver.krylov_solver
        ksp.setType("preonly")
        pc = ksp.pc
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        
        try:
            n_iter, converged = solver.solve(self._u_sol)
            
            if converged:
                print(f"  Convergence atteinte en {n_iter} itérations")
                return True
            else:
                print(f"  Pas de convergence à l'étape {step_index+1}")
                return False
                
        except RuntimeError as e:
            print(f"  Erreur lors de la résolution à l'étape {step_index+1}: {e}")
            return False
    
    def _create_point_source(self, position):
        x = ufl.SpatialCoordinate(self._domain)[0]

        epsilon = self._L/(10 * self._nb_elts)
        delta_approx = (1.0 / (epsilon * ufl.sqrt(ufl.pi))) * ufl.exp(-((x - position) / epsilon)**2)
        
        return delta_approx

    def add_point_moment(self, position:float, moment:float) -> None:
        """Ajoute un moment ponctuel à position"""
        self._pt_moment_ref.append((position, moment))

    def add_point_forces(self, pos:float, f1:float, f3:float) -> None:
        """Ajoute des forces ponctuels à position"""
        self._pt_forces_ref.append((pos, f1, f3)) 

    def set_external_loads(self, p: Optional[Union[float, Callable]] = None,
                          f1: Optional[Union[float, Callable]] = None,
                          f3: Optional[Union[float, Callable]] = None,
                          c_gamma: Optional[Union[float, Callable]] = None):
        """
        Définit les chargements distribués.
        
        Args:
            p: Pression interne
            f1: Force axiale distribuée
            f3: Force transversale distribuée
            c_gamma: Moment distribué
        """        

        if p is not None:
            if callable(p):
                self._p = p
            else:
                p_val = float(p)
                self._p = lambda x: p_val

        if f1 is not None:
            if callable(f1):
                self._f1 = f1
            else:
                f1_val = float(f1)
                self._f1 = lambda x: f1_val
    
        if f3 is not None:
            if callable(f3):
                self._f3 = f3
            else:
                f3_val = float(f3)
                self._f3 = lambda x: f3_val
        
        if c_gamma is not None:
            if callable(c_gamma):
                self._c_gamma = c_gamma
            else:
                c_gamma_val = float(c_gamma)
                self._c_gamma = lambda x: c_gamma_val


    def solve(self, load_steps=[0.1, 0.3, 0.6, 1.0]): 
        self._set_initial_geometry()
        self._apply_boundary_conditions()
                    
        # Sauvegarder les charges originales
        original_p = self._p 
        original_c_gamma = self._c_gamma

        original_f1 = self._f1
        original_f3 = self._f3

        success = True
        
        # Pour le moment on se s'occupe que de la continuation en charge pour la pression et le moment, de toute façon par sur d'utiliser f1 ou f3
        for i, factor in enumerate(load_steps):
            print(f"Étape de charge {i+1}/{len(load_steps)}: {factor*100:.0f}% de la charge")
            
            self._apply_load_factor(factor, original_p, original_c_gamma, original_f1, original_f3)

            self._setup_external_work()
            self._setup_variational_form()

            success = self._solve_step(i)

            if not success:
                break
            
        
        # Restaurer les charges originales si pb il y a eu :)
        if self._p is not None:
            self._p = original_p
        if self._c_gamma is not None:
            self._c_gamma = original_c_gamma
        
        if self._f1 is not None:
            self._p = original_f1
        if self._f3 is not None:
            self._c_gamma = original_f3

        if success:
            print("Toutes les étapes de charge ont convergé !")
        
        return success

    
    def print_data(self):
        print("\nPropriété géométrique : ")
        print(f"L = {self._L} cm")
        print(f"R = {self._R} cm")
        print(f"h = {self._h} cm")
        print(f"Eléments de types Lagrange à {self._nb_elts} éléments")

        print("\nGrandeurs comportement : ")
        print(f"E = {self._E}")
        print(f"v = {self._v}")

        print("\nEfforts extérieurs : ")
        print(f"f1 = {self._f1} N")
        print(f"f3 = {self._f3} N")
        print(f"c_gamma = {self._c_gamma} N.cm")
        print(f"p = {self._p} N/cm^2")
        
    def extract_solution_arrays(self):
        x_vals = self._V.tabulate_dof_coordinates()[:, 0]     

        dofs = [np.unique(self._V.sub(i).dofmap.list.flatten()) for i in range(4)]
        dofs_vals = [self._u_sol.x.array[dofs[i]] for i in range(4)] 
        
        sort_indices = np.argsort(x_vals)

        x_vals = x_vals[sort_indices]
        dofs_vals = [vals[sort_indices] for vals in dofs_vals]

        return x_vals, dofs_vals[0], dofs_vals[1], dofs_vals[2], dofs_vals[3]

    @property
    def radius(self):
        return self._R
