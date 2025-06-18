from inflated_beam.model import InflatedBeam, Material
from inflated_beam.visualizer import BeamVisualizer3D

import matplotlib.pyplot as plt
import numpy as np

import time

def plot_alpha_chargement(x, y):    
    fig, ax = plt.subplots()
    ax.plot(x, y, 'r.')
    ax.set_title("Evolution de $\\alpha$ à x = 75 cm en fonction du moment $c_\\gamma$")
    ax.set_xlabel("$c_\\gamma$")
    ax.set_ylabel("$\\alpha$")
    plt.show()

def beam_analysis():
    x = 75
    alpha_vals_at_x = []
    #moments = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000])
    moments = [10000]
    for moment in moments:
        material = Material(E=2e5, v=0.3)
        beam = InflatedBeam(
            L=100.0,
            R=5,
            h=0.05,  
            nb_elts=100,
            degree=2,
            material=material
        )

        beam.set_boundary_conditions(conditions_type="mine")

        beam.set_external_loads(
            p=150,
            f1=None,
            f3=None,
            c_gamma=None#lambda x: moment*x/100
        )

        beam.add_point_moment(position=100.0, moment=550000)
        beam.add_point_moment(position=0.0, moment=-550000)
        #beam.add_point_forces(pos=100, f1=-10000, f3=0)
        #beam.add_point_forces(pos=50, f1=0, f3=-100)

        sols = None
        
        start = time.time()
        if beam.solve():
            end = time.time()

            viz = BeamVisualizer3D(beam)
            viz.visualize_beam(show_both=False)
            viz.plot_cross_sections_matplotlib(positions=[0.2, 0.5, 0.8])
            viz.plot_graph_evol_dofs()
            viz.debug_solution()
            print(f"Durée d'execution : {end - start}")

            sols = beam.extract_solution_arrays()
            alpha_vals_at_x.append(sols[4][x])

        else:
            print("Echec de la résolution")
    
    #plot_alpha_chargement((moments/100), alpha_vals_at_x)
    
if __name__ == "__main__":
    print("======== ANALYSE DE POUTRE GONFLABLE ========")
    beam_analysis() 
