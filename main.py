import numpy as np
import sys

# Importez vos classes (adaptez les noms de modules)
from inflated_beam import InflatedBeam  # Remplacez par le bon nom de fichier
from beam_visualizer import BeamVisualizer3D  # Le code PyVista ci-dessus

def run_beam_analysis():
    """Lance une analyse complète de poutre avec visualisation"""
    
    print("=== ANALYSE DE POUTRE GONFLABLE ===")
    
    # Paramètres de la poutre
    L = 100.0    # longueur (cm)
    R = 5.0      # rayon initial (cm)  
    h = 0.1      # épaisseur (cm)
    nb_elts = 30 # nombre d'éléments
    degree = 2   # degré des éléments
    
    print(f"Paramètres géométriques:")
    print(f"  Longueur: {L} cm")
    print(f"  Rayon: {R} cm") 
    print(f"  Épaisseur: {h} cm")
    print(f"  Éléments: {nb_elts}")
    print(f"  Degré: {degree}")
    
    # Créer la poutre
    beam = InflatedBeam(L, R, h, nb_elts, degree)
    
    # Cas 1: Pression uniquement
    print(f"\n=== CAS 1: PRESSION SEULE ===")
    p_val = 150.0  # Pa (ajustez selon vos unités)
    print(f"Pression appliquée: {p_val} Pa")
    
    beam.set_external_loads(p=p_val, f1=None, f3=None, c_gamma=None)
    
    success = beam.solve()
    
    if success:
        print("✓ Résolution réussie")
        
        # Créer le visualisateur
        visualizer = BeamVisualizer3D(beam)
        
        # Analyser la solution
        print("\n--- Analyse de la solution ---")
        has_def = visualizer.debug_solution(beam.u_sol)
        
        if has_def:
            print("✓ Déformations détectées, génération des visualisations...")
            
            # Visualisation 3D interactive
            print("Lancement de la visualisation 3D...")
            mesh_init, mesh_def = visualizer.visualize_beam(beam.u_sol, show_both=True)
            
            # Sections transversales
            print("Génération des sections transversales...")
            fig = visualizer.plot_cross_sections_matplotlib(beam.u_sol, positions=[0.2, 0.5, 0.8])
            
        else:
            print("⚠ Aucune déformation significative détectée")
            print("Vérifiez les paramètres de charge et les conditions aux limites")
            
            # Visualiser quand même pour diagnostic
            mesh_init, mesh_def = visualizer.visualize_beam(beam.u_sol, show_both=True)
    
    else:
        print("✗ Échec de la résolution")
        return None, None
    
    # Cas 2: Pression + moment (optionnel)
    print(f"\n=== CAS 2: PRESSION + MOMENT ===")
    
    # Réinitialiser la solution
    with beam.u_sol.x.petsc_vec.localForm() as loc:
        loc.set(0.0)
    
    # Réinitialisation physique
    alpha_dofs = beam.V.sub(3).dofmap.list
    gamma_dofs = beam.V.sub(2).dofmap.list
    beam.u_sol.x.array[alpha_dofs] = 1.0
    beam.u_sol.x.array[gamma_dofs] = 0.0
    
    # Définir un moment de flexion
    def moment_function(x):
        return 5000.0 * x / L  # Moment croissant linéairement
    
    beam.set_external_loads(p=p_val*0.5, f1=None, f3=None, c_gamma=moment_function)
    
    success2 = beam.solve()
    
    if success2:
        print("✓ Résolution avec moment réussie")
        
        # Nouvelle visualisation
        visualizer2 = BeamVisualizer3D(beam)
        visualizer2.debug_solution(beam.u_sol)
        
        print("Lancement de la visualisation 3D avec moment...")
        mesh_init2, mesh_def2 = visualizer2.visualize_beam(beam.u_sol, show_both=True)
        
    else:
        print("✗ Échec de la résolution avec moment")
    
    return beam, visualizer

def diagnostic_beam_setup():
    """Diagnostic des paramètres de la poutre"""
    
    print("=== DIAGNOSTIC SETUP ===")
    
    # Test avec paramètres simples
    L, R, h = 50.0, 3.0, 0.2
    beam = InflatedBeam(L, R, h, 20, 2)
    
    print(f"Paramètres calculés:")
    print(f"  A = {beam.A:.2e}")
    print(f"  D1 = {beam.D1:.2e}")
    print(f"  D2 = {beam.D2:.2e}")
    print(f"  D3 = {beam.D3:.2e}")
    print(f"  D4 = {beam.D4:.2e}")
    
    # Test sans charge
    beam.set_external_loads(p=0, f1=0, f3=0, c_gamma=0)
    
    try:
        success = beam.solve()
        if success:
            print("✓ Résolution sans charge OK")
            visualizer = BeamVisualizer3D(beam)
            visualizer.debug_solution(beam.u_sol)
        else:
            print("✗ Échec résolution sans charge")
    except Exception as e:
        print(f"✗ Erreur: {e}")
    
    # Test avec petite pression
    beam.set_external_loads(p=100.0, f1=None, f3=None, c_gamma=None)
    
    try:
        # Réinitialiser
        with beam.u_sol.x.petsc_vec.localForm() as loc:
            loc.set(0.0)
        alpha_dofs = beam.V.sub(3).dofmap.list
        beam.u_sol.x.array[alpha_dofs] = 1.0
        
        success = beam.solve()
        if success:
            print("✓ Résolution avec petite pression OK")
            visualizer = BeamVisualizer3D(beam)
            has_def = visualizer.debug_solution(beam.u_sol)
            
            if has_def:
                print("✓ Déformations détectées avec petite pression")
                # Visualiser
                mesh_i, mesh_d = visualizer.visualize_beam(beam.u_sol)
            else:
                print("⚠ Pas de déformation avec petite pression")
                
        else:
            print("✗ Échec résolution avec petite pression")
    except Exception as e:
        print(f"✗ Erreur avec pression: {e}")

if __name__ == "__main__":
    print("Choisissez une option:")
    print("1. Analyse complète")
    print("2. Diagnostic simple")
    
    choice = input("Votre choix (1 ou 2): ").strip()
    
    if choice == "1":
        beam, visualizer = run_beam_analysis()
    elif choice == "2":
        diagnostic_beam_setup()
    else:
        print("Lancement du diagnostic par défaut...")
        diagnostic_beam_setup()
    
    print("\nAnalyse terminée.")
