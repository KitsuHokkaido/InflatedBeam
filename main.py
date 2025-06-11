from inflated_beam import InflatedBeam
from beam_visualizer import BeamVisualizer


if __name__ == '__main__':
   beam = InflatedBeam(L=10.0, R=3, h=0.01, nb_elts=50, degree=3) 
   
   beam.set_external_loads(
        p=1000.0,
        f1=0,
        f3=0,
        c_gamma=200.0
   )

   if beam.solve():
       u1, u3, gamma, alpha = beam.extract_solution()

   visualizer = BeamVisualizer()

   domain, tdim = beam.get_data()

   visualizer.show(domain, tdim)


