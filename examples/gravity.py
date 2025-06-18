from inflated_beam.model import InflatedBeam, Material
from inflated_beam.visualizer import BeamVisualizer3D

import time

def beam_analysis():
    material = Material(E=2e5, v=0.3)
    beam = InflatedBeam(
        L=100.0,
        R=5,
        h=0.05,  
        nb_elts=100,
        degree=2,
        material=material
    )

    beam.set_boundary_conditions(conditions_type='left_clamped')
    
    beam.set_external_loads(
        p=150,
        f1=None,
        f3=-100,
        c_gamma=None
    )

    start = time.time()
    if beam.solve():
        end = time.time()
        
        viz = BeamVisualizer3D(beam)
        viz.visualize_beam(show_both=False)
        viz.plot_cross_sections_matplotlib()
        viz.plot_graph_evol_dofs()
        viz.debug_solution()
        
        print(f"Durée d'execution : {end - start}")


if __name__ == '__main__':
    beam_analysis()
