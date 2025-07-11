from inflated_beam.model import InflatedBeam, Material
from inflated_beam.visualizer import BeamVisualizer3D

import time

def beam_analysis():
    material = Material(E=400, v=0.3)
    beam = InflatedBeam(
        L=60.0,
        R=3.0,
        h=0.05,  
        nb_elts=100,
        degree=2,
        material=material
    )

    beam.set_initial_geometry(geometry_type='cylinder')
    beam.set_boundary_conditions(conditions_type='simply_supported')
    
    beam.set_external_loads(
        p=12,
        f1=None,
        f3=None,
        c_gamma=None
    )
    
    beam.add_point_forces(pos=30, f1=0, f3=-100)
    #beam.add_point_forces(pos=60.0, f1=-20, f3=0)

    start = time.time()
    load_steps = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if beam.solve(load_steps):
        end = time.time()
        
        viz = BeamVisualizer3D(beam)
        viz.visualize_beam(show_both=False)
        viz.plot_cross_sections_matplotlib(positions=[0, 0.5, 1])
        viz.plot_graph_evol_dofs()
        viz.plot_lagrange_constraint()
        viz.debug_solution()
        
        print(f"Durée d'execution : {end - start}")


if __name__ == '__main__':
    beam_analysis()

