from inflated_beam import InflatedBeam
from beam_visualizer import BeamVisualizer3D

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
    moments = [6000]
    for moment in moments:
        beam = InflatedBeam(
            L=100.0,
            R=5,
            h=0.01,  
            nb_elts=300,
            degree=3
        )

        beam.set_external_loads(
            p=150,
            f1=None,
            f3=-2,
            c_gamma=None#lambda x: moment*x/100
        )

        #beam.add_point_moment(position=100.0, moment=6000)
        #beam.add_point_forces(pos=100, f1=-6000, f3=0)
        #beam.add_point_forces(pos=50, f1=0, f3=-1000)

        sols = None
        
        start = time.time()
        if beam.solve():
            end = time.time()

            viz = BeamVisualizer3D(beam)
            #viz.visualize_beam(show_both=False)
            #viz.plot_cross_sections_matplotlib()
            #viz.plot_graph_evol_dofs()
            viz.debug_solution()
            print(f"Durée d'execution : {end - start}")

            sols = viz.extract_solution_arrays()
            alpha_vals_at_x.append(sols[4][x])

        else:
            print("Echec de la résolution")

    #plot_alpha_chargement((moments/100), alpha_vals_at_x)
    
if __name__ == "__main__":
    print("======== ANALYSE DE POUTRE GONFLABLE ========")
    beam_analysis() 
