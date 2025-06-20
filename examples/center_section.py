from inflated_beam.model import InflatedBeam, Material
from inflated_beam.visualizer import BeamVisualizer3D

def run_beam_analysis():
    material = Material(E=2e5, v=0.3)
    beam = InflatedBeam(
        L=100.0,
        R=5,
        h=0.05,  
        nb_elts=100,
        degree=2,
        material=material
    )

    beam.set_boundary_conditions(conditions_type='center_deformation')

    beam.set_external_loads(
        p=150,
        f1=None,
        f3=None,
        c_gamma=None
    )

    beam.add_point_moment(50, 200000)

    if beam.solve():
        viz = BeamVisualizer3D(beam)
        viz.visualize_beam(show_both=False)
        viz.plot_graph_evol_dofs()

if __name__ == '__main__':
    run_beam_analysis()
