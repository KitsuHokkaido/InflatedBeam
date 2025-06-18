# Inflated Beam

Ce projet a pour objectif de déterminer les déformations d'une poutre gonflable.

## Installation

### Le projet

Il faut tout d'abord cloner le dépôt 

```bash
cd ~/.local/dev
git clone [nom_du_repo]
```

Il est conseillé de créer un environnement conda 
```bash
conda create -n inflatedbeam-env
conda activate inflatedbeam-env
```

Puis à la racine du dépot, lancer l'installation
```bash
cd InflatedBeam/
conda install pip
pip install -e .
```

### Dépendances

Des dépendances sont nécessaires pour faire fonctionner la bib qui s'appuie sur Fenics

```bash
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

## Exemple d'utilisation

Des exemples sont disponibles dans le dossier examples/. Voici un exemple minimaliste requis pour que la bib fonctionne :

```python
from inflated_beam.visualizer import BeamVisualizer3D
from inflated_beam.model import InflatedBeam, Material

material = Material(E=2e5, v=0.3)

beam = InflatedBeam(L=100.0, R=5, h=0.05, nb_elts=100, degree=2, material=material)
beam.set_boundary_conditions(conditions_type='left_clamped')
beam.set_external_loads(p=150, f1=10, f3=None, c_gamma=None)

if beam.solve():
    viz = BeamVisualizer3D(beam)            
    viz.visualize_beam()
```
