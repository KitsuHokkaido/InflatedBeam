from inflated_beam.model import InflatedBeam, Material
from inflated_beam.visualizer import BeamVisualizer3D

def run_beam_analysis(critere_convergence=1e-2):    
    former_moment = 0
    current_moment = 13000
    epsilon = 100

    while True:
        material = Material(E=200, v=0.3)
        beam = InflatedBeam(
            L=100.0,
            R=5,
            h=0.05,  
            nb_elts=100,
            degree=2,
            material=material
        )
        print(f"\nLancement d'une nouvelle tentative pour c = {current_moment}")
        print("="*50)
        beam.set_initial_geometry(geometry_type='imperfection', position=50, ddl=3, imperfection_amplitude=0.3)
        beam.set_boundary_conditions(conditions_type='buckling')

        beam.set_external_loads(
            p=13,
            f1=None,
            f3=None,
            c_gamma=None
        )

        beam.add_point_moment(position=100.0, moment=current_moment)
        beam.add_point_moment(position=0.0, moment=-current_moment)
        #beam.add_point_forces(pos=100, f1=-10, f3=0)

        if beam.solve():
            print(f"\nLa solution a convergé pour c = {current_moment} N/cm")
            former_moment = current_moment
            current_moment += epsilon

            if epsilon < critere_convergence:
                print("\n")
                print("#"*36)
                print("# Critère de convergence atteint   #\n# Lancement de la visualisation... #")             
                print("#"*36)
                print("\n")

                viz = BeamVisualizer3D(beam)
                viz.visualize_beam(show_both=False)
                viz.plot_graph_evol_dofs()
                viz.plot_lagrange_constraint()

                break

        else:
            print(f"\nLa solution n'a pas convergé pour c = {current_moment} N/cm")
            epsilon = (current_moment - former_moment)/ 2
            current_moment = former_moment + epsilon
           
        
if __name__ == '__main__':
    run_beam_analysis(critere_convergence=1)
