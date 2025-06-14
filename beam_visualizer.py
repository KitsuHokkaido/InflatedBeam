import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

class BeamVisualizer3D:
    def __init__(self, beam):
        self.beam = beam
       
    def extract_solution_arrays(self, n_points=100):
        x_eval = np.linspace(0, self.beam.L, n_points)
        
        u1_vals = np.zeros_like(x_eval)
        u3_vals = np.zeros_like(x_eval)
        gamma_vals = np.zeros_like(x_eval)
        alpha_vals = np.zeros_like(x_eval)
        
        # Évaluer les solutions en chaque point
        for i, x in enumerate(x_eval):
            point = np.array([[x, 0.0, 0.0]])
            try:
                # Évaluer la solution complète au point x
                u_val = self.beam.u_sol.eval(point, np.array([0]))
                u1_vals[i] = u_val[0]
                u3_vals[i] = u_val[1] 
                gamma_vals[i] = u_val[2]
                alpha_vals[i] = u_val[3]
            except Exception as e:
                print(f"Erreur d'évaluation au point {x}: {e}")
                # Si l'évaluation échoue, utiliser les valeurs précédentes ou valeurs par défaut
                if i > 0:
                    u1_vals[i] = u1_vals[i-1]
                    u3_vals[i] = u3_vals[i-1]
                    gamma_vals[i] = gamma_vals[i-1]
                    alpha_vals[i] = alpha_vals[i-1]
                else:
                    # Valeurs initiales
                    u1_vals[i] = 0.0
                    u3_vals[i] = 0.0
                    gamma_vals[i] = 0.0
                    alpha_vals[i] = 1.0
        
        return x_eval, u1_vals, u3_vals, gamma_vals, alpha_vals
    
    def compute_3d_geometry(self, x_vals, u1_vals, u3_vals, gamma_vals, alpha_vals, n_theta=32):
        """Calcule la géométrie 3D de la poutre déformée"""
        
        # Paramètre angulaire pour l'ellipse
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        
        # Nombre de points
        n_x = len(x_vals)
        
        # Initialiser les tableaux 3D
        points = []
        
        for i, x1 in enumerate(x_vals):
            x_center = x1 + u1_vals[i]  # Coordonnée X du centre
            y_center = 0.0              # Pas de déplacement en Y pour le centre
            z_center = u3_vals[i]       # Déplacement transversal du centre
            
            # Paramètres de l'ellipse
            alpha = alpha_vals[i]
            gamma = gamma_vals[i]
            
            # Demi-axes de l'ellipse 
            a = self.beam.R * alpha
            b = self.beam.R * np.sqrt(max(0.01, 2 - alpha**2))  # Protection contre les valeurs négatives
            
            # Base locale tournée par gamma 
            cos_gamma = np.cos(gamma)
            sin_gamma = np.sin(gamma)
            
            for th in theta:
                # Coordonnées paramétriques de l'ellipse dans la base locale
                y_local = a * np.cos(th)
                z_local = b * np.sin(th)
                
                # Projection dans la base globale
                proj_e3y_e1o = - z_local * sin_gamma
                proj_e3y_e3o = z_local * cos_gamma
                
                # Position finale : OM = OG + GM
                x_final = x_center + proj_e3y_e1o 
                y_final = y_center + y_local
                z_final = z_center + proj_e3y_e3o
                
                points.append([x_final, y_final, z_final])
        
        points = np.array(points)
        return points.reshape(n_x, n_theta, 3)
    
    def compute_initial_geometry(self, x_vals, n_theta=32):
        """Calcule la géométrie 3D de la poutre non déformée (cylindre)"""
        
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        n_x = len(x_vals)
        
        points = []
        
        for i, x1 in enumerate(x_vals):
            for j, th in enumerate(theta):
                # Cylindre circulaire non déformé
                x_final = x1
                y_final = self.beam.R * np.cos(th)
                z_final = self.beam.R * np.sin(th)
                
                points.append([x_final, y_final, z_final])
                
        points = np.array(points)
        return points.reshape(n_x, n_theta, 3)
    
    def create_mesh_from_points(self, points_3d):
        """Crée un maillage PyVista à partir des points 3D"""
        n_x, n_theta, _ = points_3d.shape
        
        # Aplatir les points
        points = points_3d.reshape(-1, 3)
        
        # Créer les faces (quads)
        faces = []
        
        for i in range(n_x - 1):
            for j in range(n_theta):
                # Indices des 4 coins du quad
                p1 = i * n_theta + j
                p2 = i * n_theta + (j + 1) % n_theta
                p3 = (i + 1) * n_theta + (j + 1) % n_theta
                p4 = (i + 1) * n_theta + j
                
                # Ajouter le quad (4 points + 4 indices)
                faces.extend([4, p1, p2, p3, p4])
        
        # Créer le maillage PyVista
        mesh = pv.PolyData(points, faces)
        return mesh
    
    def visualize_beam(self, show_both=True, window_size=(1200, 600)):
        """Visualise la poutre avec PyVista"""
        
        # Extraire les solutions
        x_vals, u1_vals, u3_vals, gamma_vals, alpha_vals = self.extract_solution_arrays(n_points=50)
        
        # Afficher quelques valeurs pour diagnostic
        print(f"Valeurs de déformation détectées:")
        print(f"  u1 max: {np.max(np.abs(u1_vals)):.6f}")
        print(f"  u3 max: {np.max(np.abs(u3_vals)):.6f}")
        print(f"  gamma max: {np.max(np.abs(gamma_vals)):.6f}")
        print(f"  alpha min/max: {np.min(alpha_vals):.6f} / {np.max(alpha_vals):.6f}")
        
        # Calculer les géométries
        points_init = self.compute_initial_geometry(x_vals)
        points_def = self.compute_3d_geometry(x_vals, u1_vals, u3_vals, gamma_vals, alpha_vals)
        
        # Créer les maillages
        mesh_init = self.create_mesh_from_points(points_init)
        mesh_def = self.create_mesh_from_points(points_def)
        
        if show_both:
            # Visualisation comparative
            plotter = pv.Plotter(shape=(1, 2), window_size=window_size)
            
            # Configuration initiale
            plotter.subplot(0, 0)
            plotter.add_mesh(mesh_init, color='lightblue', opacity=0.8, show_edges=True)
            plotter.add_title('Configuration initiale')
            plotter.show_axes()
            
            # Configuration déformée
            plotter.subplot(0, 1)
            plotter.add_mesh(mesh_def, color='lightcoral', opacity=0.8, show_edges=True)
            plotter.add_title('Configuration déformée')
            plotter.show_axes()
            
        else:
            # Visualisation superposée
            plotter = pv.Plotter(window_size=window_size)
            plotter.add_mesh(mesh_init, color='lightblue', show_edges=True, opacity=0.3, label='Initial')
            plotter.add_mesh(mesh_def, color='lightcoral', show_edges=True, opacity=0.8, label='Déformé')
            plotter.add_title('Comparaison des configurations')
            plotter.add_legend()
            plotter.show_axes()
        
        plotter.show()
        
        return mesh_init, mesh_def
    
    def plot_cross_sections_matplotlib(self, positions=[0.25, 0.5, 1]):
        """Trace les sections transversales avec matplotlib pour vérification"""
        
        x_vals, u1_vals, u3_vals, gamma_vals, alpha_vals = self.extract_solution_arrays(n_points=50)
        
        fig, axes = plt.subplots(1, len(positions), figsize=(4*len(positions), 4))
        if len(positions) == 1:
            axes = [axes]
        
        theta = np.linspace(0, 2*np.pi, 100)

        index = ['a', 'b', 'c']
        
        for i, pos_ratio in enumerate(positions):
            # Trouver l'indice correspondant à la position
            idx = int(pos_ratio * (len(x_vals) - 1))
            x_pos = x_vals[idx]
            
            # Paramètres de l'ellipse à cette position
            alpha = alpha_vals[idx]
            gamma = gamma_vals[idx]
            
            print(f"Position {pos_ratio}: alpha={alpha:.4f}, gamma={gamma:.4f}")
            
            # Demi-axes
            a = self.beam.R * alpha
            b = self.beam.R * np.sqrt(max(0.01, 2 - alpha**2))
            
            # Ellipse dans le repère local
            y_local = a * np.cos(theta)
            z_local = b * np.sin(theta)
            
            axes[i].plot(y_local, z_local, 'r-', linewidth=2, label='Déformée')
            
            # Cercle initial pour comparaison
            y_circle = self.beam.R * np.cos(theta)
            z_circle = self.beam.R * np.sin(theta)
            axes[i].plot(y_circle, z_circle, 'b--', linewidth=1, alpha=0.7, label='Initiale')
        
            axes[i].set_aspect('equal')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'(α = {alpha:.3f}, γ = {gamma:.3f} rad)')
            axes[i].set_xlabel('y (cm)')
            axes[i].text(0.5, -0.25, f"({index[i]}) Section à x = {x_pos:.1f} cm", transform=axes[i].transAxes, ha='center', fontsize=12)
            axes[i].set_ylabel('z (cm)')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()
        return fig

    def plot_graph_evol_dofs(self):
        data = self.extract_solution_arrays(n_points=50)
        x_vals = data[0]
        dofs_vals = data[1:]

        titles = ['Déplacement longitudinal', 'Déplacement transversal', 
              'Rotation de section', 'Paramètre de forme']
        
        ylabels = [r'$u_1$ (cm)', r'$u_3$ (cm)', r'$\gamma$ (rad)', r'$\alpha$ (-)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig, axes = plt.subplots(2, int(len(dofs_vals)/2), figsize=(4*(len(dofs_vals)/2), 4*(len(dofs_vals)/2)))
        fig.suptitle('Évolution des variables le long de la poutre', fontsize=16, fontweight='bold') 
        
        for i in range(2):
            for j in range(2):
                index = 2 * i + j

                axes[i, j].plot(x_vals, dofs_vals[index], color=colors[index], linewidth=1.5, alpha=0.9, label=ylabels[index])
                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_title(titles[index], pad=15)
                axes[i, j].set_xlabel(r'$x_1$ (cm)')
                axes[i, j].set_ylabel(ylabels[index])

        plt.tight_layout()
        plt.show()
            
    
    def debug_solution(self):
        """Fonction de debug pour vérifier les valeurs de la solution"""
        x_vals, u1_vals, u3_vals, gamma_vals, alpha_vals = self.extract_solution_arrays(n_points=50)
        
        print("=== DEBUG SOLUTION ===")
        self.beam.print_data()
        print("")
        print(f"Nombre de points d'évaluation: {len(x_vals)}")
        print(f"Range x: [{x_vals[0]:.2f}, {x_vals[-1]:.2f}]")
        
        print("\nStatistiques des variables:")
        variables = [
            ("u1 (dépl. axial)", u1_vals),
            ("u3 (dépl. transv.)", u3_vals), 
            ("gamma (rotation)", gamma_vals),
            ("alpha (forme)", alpha_vals)
        ]
        
        for name, vals in variables:
            print(f"{name:20s}: min={np.min(vals):8.4f}, max={np.max(vals):8.4f}, "
                  f"mean={np.mean(vals):8.4f}, std={np.std(vals):8.4f}")
        
        # Vérifier si il y a des variations significatives
        has_deformation = (
            np.std(u1_vals) > 1e-6 or 
            np.std(u3_vals) > 1e-6 or 
            np.std(gamma_vals) > 1e-6 or 
            np.std(alpha_vals) > 1e-6
        )
        
        print(f"\nDéformation détectée: {'OUI' if has_deformation else 'NON'}")
        
        return has_deformation

