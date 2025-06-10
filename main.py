from inflated_beam import InflatedBeam
from visualizer import Visualizer


if __name__ == '__main__':
   beam = InflatedBeam(L=10.0, R=3, h=0.01, nb_elts=50, degree=3) 
   visualizer = Visualizer()

   domain, tdim = beam.get_data()

   visualizer.show(domain, tdim)


