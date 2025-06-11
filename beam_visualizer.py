from dolfinx import plot
import pyvista 


class BeamVisualizer:
    def __init__(self):
        self.plotter = pyvista.Plotter() 
    
    def show(self, domain, tdim):
        topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        self.plotter.add_mesh(grid, show_edges=True)
        self.plotter.show_axes()

        if not pyvista.OFF_SCREEN:
            self.plotter.show()
    
