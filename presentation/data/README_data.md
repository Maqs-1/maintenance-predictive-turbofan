# Dataset — NASA C-MAPSS FD002

Le dataset n'est pas inclus dans ce repository (fichiers trop volumineux pour GitHub).

## Téléchargement

### Option 1 — Kaggle (recommandé, utilisé dans le notebook)

```bash
pip install kagglehub
```

Le notebook télécharge automatiquement le dataset via `kagglehub` à l'étape 1 :

```python
import kagglehub
raw_path = kagglehub.dataset_download('behrad3d/nasa-cmaps')
```

Une authentification Kaggle est nécessaire. Créez un compte sur [kaggle.com](https://www.kaggle.com) et configurez votre clé API :

```bash
# Placez votre kaggle.json dans ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Option 2 — UCI Machine Learning Repository

Téléchargement direct : https://archive.ics.uci.edu/ml/datasets/Turbofan+Engine+Degradation+Simulation

Placer les fichiers dans ce dossier `data/` avec la structure suivante :

```
data/
└── CMaps/
    ├── train_FD001.txt
    ├── train_FD002.txt   ← utilisé dans ce projet
    ├── test_FD001.txt
    ├── test_FD002.txt    ← utilisé dans ce projet
    ├── RUL_FD001.txt
    └── RUL_FD002.txt     ← utilisé dans ce projet
```

## Description du dataset FD002

| Fichier | Description | Lignes |
|---|---|---|
| train_FD002.txt | Données d'entraînement — moteurs jusqu'à panne | 53 759 |
| test_FD002.txt | Données de test — derniers cycles observés | 33 991 |
| RUL_FD002.txt | Ground truth RUL pour chaque moteur test | 259 |

## Pourquoi FD002 ?

FD002 est le sous-dataset le plus réaliste avec 6 conditions de vol simultanées (vs 1 pour FD001).
C'est la référence académique standard pour comparer les algorithmes de pronostic de pannes.

**Référence :** Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
Damage propagation modeling for aircraft engine run-to-failure simulation.
*International Conference on Prognostics and Health Management (PHM 2008).*
