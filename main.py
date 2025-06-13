from inflated_beam import InflatedBeam
from beam_visualizer import BeamVisualizer3D

import matplotlib.pyplot as plt
import numpy as np

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
    moments = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000])
    for moment in moments:
        beam = InflatedBeam(
            L=100.0,
            R=5,
            h=0.1,  
            nb_elts=20,
            degree=2
        )

        beam.set_external_loads(
            p=150,
            f1=0,
            f3=0,
            c_gamma=lambda x:moment*x/100
        )

        sols = None

        if(beam.solve()):
            viz = BeamVisualizer3D(beam)
            viz.visualize_beam(show_both=False)
            viz.plot_cross_sections_matplotlib()
            viz.plot_graph_evol_dofs()
            viz.debug_solution()

            sols = viz.extract_solution_arrays()
            alpha_vals_at_x.append(sols[4][x])

        else:
            print("Echec de la résolution")

    #plot_alpha_chargement((moments/100), alpha_vals_at_x)
    
if __name__ == "__main__":
    print("======== ANALYSE DE POUTRE GONFLABLE ========")
    beam_analysis() 
