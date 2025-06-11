from inflated_beam import InflatedBeam
from beam_visualizer import BeamVisualizer


if __name__ == '__main__':
   # Cas doux en cm
   beam_test = InflatedBeam(
        L=30.0,      
        R=3.0,       
        h=0.2,       # Plus épais (2 mm)
        nb_elts=20,  # Moins d'éléments pour commencer
        degree=2,
        E=200000,    # Module plus faible
        v=0.3
   )

   beam_test.set_external_loads(
        p=50.0,      # Pression plus faible
        f1=0,
        f3=0,
        c_gamma=5.0  # Moment plus faible
   )

   if beam_test.solve():
       u1, u3, gamma, alpha, lam = beam_test.extract_solution()

   #visualizer = BeamVisualizer()

   #domain, tdim = beam.get_data()

   #visualizer.show(domain, tdim)


