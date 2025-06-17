import numpy as np
import ufl

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type

class InflatedBeam:
    def __init__(self, L, R, h, nb_elts, degree, E=2e5, v=0.3):
        self.degree = degree
        self.nb_elts = nb_elts
        
        # Grandeurs du systeme (en cm)
        self.L = L
        self.R = R
        self.h = h

        # Grandeurs comportements
        self.E = E
        self.v = v

        self.A = E*h
        self.D1 = E*h**3/12*(1 - v**2)
        self.D2 = self.D1
        self.D3 = v*self.D1
        self.D4 = E*h**3/24*(1 + v**2)

        # Efforts extérieurs
        self.p = None
        self.f1 = None
        self.f3 = None
        self.c_gamma = None
        
        # Efforts extérieurs en un pt 
        self.point_moment = []
        self.point_forces = []

        self.pt_moment_ref = []
        self.pt_forces_ref = []

        self.deltas = []

        self.domain = mesh.create_interval(MPI.COMM_WORLD, nb_elts, [0.0, L])
        self.tdim = self.domain.topology.dim

        self.setup_function_space()
    
    def setup_function_space(self):
        self.V = fem.functionspace(self.domain, ("Lagrange", self.degree, (5,)))

        self.v_test = ufl.TestFunction(self.V)   
        self.v1, self.v3, self.v_gamma, self.v_alpha, self.v_lam = ufl.split(self.v_test)
        
        self.u_sol = fem.Function(self.V)
        self.u1_sol, self.u3_sol, self.gamma_sol, self.alpha_sol, self.lam_sol = ufl.split(self.u_sol)
    
    def setup_variational_form(self):
        u1_1 = self.u1_sol.dx(0)
        u3_1 = self.u3_sol.dx(0)
        alpha_1 = self.alpha_sol.dx(0)
        alpha_11 = alpha_1.dx(0)

        e_p = u1_1 + 0.5 * (u1_1**2 + u3_1**2) 
        k_p = self.gamma_sol.dx(0)
        k_22s = self.R*(self.alpha_sol*ufl.sqrt(2 - self.alpha_sol**2) - 1)

        H = ( 2 * alpha_1**2 + alpha_11*(2*self.alpha_sol + self.alpha_sol**3) ) / (2 - self.alpha_sol**2)**(3/2)

        overline_es_squared = (self.R**2 * alpha_1**2 * ufl.pi ) / (2 - self.alpha_sol**2)

        overline_k11s_squared = (self.R**2*ufl.pi/4) * (3*(2 - self.alpha_sol**2)*(alpha_11)**2 - 2*ufl.sqrt(2 - self.alpha_sol**2)*alpha_11*H + 3*self.alpha_sol**2*H**2)

        overline_k21s_squared = (self.R**2 * alpha_1**2 * ufl.pi) / (2 - self.alpha_sol**2)

        overline_k11s = self.R * ufl.pi * (self.alpha_sol*H - alpha_11*ufl.sqrt(2 - self.alpha_sol**2))

        U_p = (self.R/2) * (
            2*ufl.pi * self.A * e_p**2 + 
            (self.A * self.R**2 * (2 - self.alpha_sol**2) * ufl.pi + 
             self.D1 * self.alpha_sol**2 * ufl.pi) * k_p**2
        )
        
        U_s = (self.R/2) * (
            self.A * overline_es_squared + 
            self.D1 * overline_k11s_squared + 
            self.D2 * k_22s**2 + 
            4*self.D4 * overline_k21s_squared + 
            2*self.D3 * k_22s * overline_k11s
        )
        
        U_ps = self.R * self.A * ufl.pi * e_p * self.R**2 * (alpha_1**2) / (2 - self.alpha_sol**2)
        
        energy_density = U_p + U_s + U_ps
        self.total_internal_energy = energy_density * ufl.dx
        
        self.setup_lagrange_multiplier()
        
        self.total_potential = self.total_internal_energy - self.total_external_work + self.lagrange_term 
        
        self.F = ufl.derivative(self.total_potential, self.u_sol, self.v_test)

    def setup_lagrange_multiplier(self):
        u1_1 = self.u1_sol.dx(0)
        u3_1 = self.u3_sol.dx(0)
        j_p = ufl.sqrt((1 + u1_1)**2 + u3_1**2)

        C = - ufl.sin(self.gamma_sol)*(1 + u1_1)/j_p - ufl.cos(self.gamma_sol)*u3_1/j_p
    
        constraint_term_density = self.lam_sol * C
        self.lagrange_term = constraint_term_density * ufl.dx

    def setup_external_work(self):

        x = ufl.SpatialCoordinate(self.domain)[0]
        
        if self.p is None:
            p = 0.0
        else:
            p = self.p(x)

        if self.f1 is None:
            f1 = 0
        else:
            f1 = self.f1(x)

        if self.f3 is None:
            f3 = 0
        else:
            f3 = self.f3(x)

        if self.c_gamma is None:
            c_gamma = 0
        else:
            c_gamma = self.c_gamma(x)

        
        pressure_work = p*ufl.pi*self.R**2*(self.alpha_sol*ufl.cos(self.gamma_sol) + ufl.sqrt(2 - self.alpha_sol**2)*(1 - 2*self.alpha_sol))
        force_work = f1*self.u1_sol + f3*self.u3_sol
        moment_work = c_gamma*self.gamma_sol

        point_moment_work = 0
        point_force_work = 0

        for pos, moment in self.point_moment:
            delta = self.create_point_source(pos)
            point_moment_work += moment * self.gamma_sol * delta

        for pos, f1, f3 in self.point_forces:
            delta = self.create_point_source(pos)
            point_force_work += f1 * self.u1_sol * delta + f3 * self.u3_sol * delta
            self.deltas.append((pos, delta))

        external_work_density = pressure_work + force_work + moment_work + point_moment_work + point_force_work
        self.total_external_work = external_work_density * ufl.dx

    def set_external_loads(self, p, f1, f3, c_gamma):
        if p is not None:
            if callable(p):
                self.p = p
            else:
                p_val = float(p)
                self.p = lambda x: p_val

        if f1 is not None:
            if callable(f1):
                self.f1 = f1
            else:
                f1_val = float(f1)
                self.f1 = lambda x: f1_val
    
        if f3 is not None:
            if callable(f3):
                self.f3 = f3
            else:
                f3_val = float(f3)
                self.f3 = lambda x: f3_val
        
        if c_gamma is not None:
            if callable(c_gamma):
                self.c_gamma = c_gamma
            else:
                c_gamma_val = float(c_gamma)
                self.c_gamma = lambda x: c_gamma_val


    def apply_boundary_conditions(self):
        def boundary_left(x):
            return np.isclose(x[0], 0.0)

        def boundary_right(x):
            return np.isclose(x[0], self.L)

        boundary_facets_left = mesh.locate_entities_boundary(self.domain, self.tdim-1, boundary_left)
        boundary_facets_right = mesh.locate_entities_boundary(self.domain, self.tdim-1, boundary_right)
        
        # Pour chaque composante
        dofs_u1 = fem.locate_dofs_topological(self.V.sub(0), self.tdim-1, boundary_facets_left)
        dofs_u3 = fem.locate_dofs_topological(self.V.sub(1), self.tdim-1, boundary_facets_left)  
        dofs_gamma = fem.locate_dofs_topological(self.V.sub(2), self.tdim-1, boundary_facets_left)
        dofs_alpha = fem.locate_dofs_topological(self.V.sub(3), self.tdim-1, boundary_facets_left)
        
        bc_u1 = fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(0.0)), dofs_u1, self.V.sub(0))
        bc_u3 = fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(0.0)), dofs_u3, self.V.sub(1))
        bc_gamma = fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(0.0)), dofs_gamma, self.V.sub(2))
        bc_alpha = fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(1.0)), dofs_alpha, self.V.sub(3))


        dofs_u3_right = fem.locate_dofs_topological(self.V.sub(1), self.tdim-1, boundary_facets_right)
        dofs_gamma_right = fem.locate_dofs_topological(self.V.sub(2), self.tdim-1, boundary_facets_right)
        dofs_alpha_right = fem.locate_dofs_topological(self.V.sub(3), self.tdim-1, boundary_facets_right)
        
        bc_u3_right = fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(0.0)), dofs_u3_right, self.V.sub(1))
        bc_gamma_right = fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(-np.pi/6)), dofs_gamma_right, self.V.sub(2))
        bc_alpha_right = fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(1.0)), dofs_alpha_right, self.V.sub(3))
        
        self.bcs = [bc_u1, bc_u3, bc_alpha, bc_u3_right, bc_alpha_right]

    def set_initial_geometry(self):
        # Initialisation de la solution avec des valeurs physiques
        with self.u_sol.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        
        # On récup ici l'ensemble des positions des champs alpah et gamma au niveau des noeuds 
        alpha_dofs = self.V.sub(3).dofmap.list
        gamma_dofs = self.V.sub(2).dofmap.list
        
        # et ici on applique donc à tous les noeuds
        self.u_sol.x.array[alpha_dofs] = 1.0 
        self.u_sol.x.array[gamma_dofs] = 0.0


    def solve(self, load_steps=[0.1, 0.3, 0.6, 1.0]): 
        self.set_initial_geometry()
        self.apply_boundary_conditions()
                    
        # Sauvegarder les charges originales
        original_p = self.p 
        original_c_gamma = self.c_gamma

        original_f1 = self.f1
        original_f3 = self.f3

        success = True
        
        # Pour le moment on se s'occupe que de la continuation en charge pour la pression et le moment, de toute façon par sur d'utiliser f1 ou f3
        for i, factor in enumerate(load_steps):
            print(f"Étape de charge {i+1}/{len(load_steps)}: {factor*100:.0f}% de la charge")
            
            self.apply_load_factor(factor, original_p, original_c_gamma, original_f1, original_f3)

            self.setup_external_work()
            self.setup_variational_form()

            success = self.solve_step(i)

            if not success:
                break
            
        
        # Restaurer les charges originales si pb il y a eu :)
        if self.p is not None:
            self.p = original_p
        if self.c_gamma is not None:
            self.c_gamma = original_c_gamma
        
        if self.f1 is not None:
            self.p = original_f1
        if self.f3 is not None:
            self.c_gamma = original_f3

        if success:
            print("Toutes les étapes de charge ont convergé !")
        
        return success

    def apply_load_factor(self, factor, original_p, original_c_gamma, original_f1, original_f3):    
        # Appliquer le facteur de charge
        if self.p is not None:
            self.p = lambda x: factor * original_p(x)
        if self.c_gamma is not None:
            self.c_gamma = lambda x: factor * original_c_gamma(x)

        if self.f1 is not None:
            self.f1 = lambda x: factor * original_f1(x)
        if self.f3 is not None:
            self.f3 = lambda x: factor * original_f3(x)
       
        # Appliquer le facteur de charge aux points sources
        self.point_moment = [(pos, moment * factor) for pos, moment in self.pt_moment_ref]
    
        self.point_forces = [(pos, f1 * factor, f3 * factor) for pos, f1, f3 in self.pt_forces_ref]


    def solve_step(self, step_index): 
        problem = NonlinearProblem(self.F, self.u_sol, self.bcs)
        solver = NewtonSolver(self.domain.comm, problem)
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
            n_iter, converged = solver.solve(self.u_sol)
            
            if converged:
                print(f"  Convergence atteinte en {n_iter} itérations")
                return True
            else:
                print(f"  Pas de convergence à l'étape {step_index+1}")
                return False
                
        except RuntimeError as e:
            print(f"  Erreur lors de la résolution à l'étape {step_index+1}: {e}")
            return False

    def add_point_moment(self, position, moment):
        self.pt_moment_ref.append((position, moment))

    def add_point_forces(self, pos, f1, f3):
        self.pt_forces_ref.append((pos, f1, f3)) 

    def create_point_source(self, position):
        x = ufl.SpatialCoordinate(self.domain)[0]

        epsilon = self.L/(10 * self.nb_elts)
        delta_approx = (1.0 / (epsilon * ufl.sqrt(ufl.pi))) * ufl.exp(-((x - position) / epsilon)**2)
        
        return delta_approx
    
    def print_data(self):
        print("\nPropriété géométrique : ")
        print(f"L = {self.L} cm")
        print(f"R = {self.R} cm")
        print(f"h = {self.h} cm")
        print(f"Eléments de types Lagrange à {self.nb_elts} éléments")

        print("\nGrandeurs comportement : ")
        print(f"E = {self.E}")
        print(f"v = {self.v}")

        print("\nEfforts extérieurs : ")
        print(f"f1 = {self.f1} N")
        print(f"f3 = {self.f3} N")
        print(f"c_gamma = {self.c_gamma} N.cm")
        print(f"p = {self.p} N/cm^2")

        print("\nPt source : ")
        for pos, delta in self.deltas:
            print(f"{pos} : {delta}")

       
