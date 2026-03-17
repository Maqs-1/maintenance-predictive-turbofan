# Maintenance Prédictive des Moteurs Turbofan

**Prédiction de la Remaining Useful Life (RUL) — NASA C-MAPSS FD002**

DU Data Analytics · Université Paris 1 Panthéon-Sorbonne · Projet Machine Learning

---

## Le problème

Un moteur d'avion se dégrade progressivement et inévitablement. Aujourd'hui, deux approches coexistent pour gérer cette dégradation : la maintenance corrective (on répare après la panne) et la maintenance préventive (on change à intervalles fixes). Les deux sont coûteuses et imprécises — l'une met en danger la sécurité, l'autre gaspille des ressources.

Ce projet répond à une question concrète : **peut-on prédire précisément combien de cycles de vol restent à un moteur avant sa panne ?**

---

## La solution

Un pipeline Machine Learning complet qui prédit le **RUL (Remaining Useful Life)** — le nombre de cycles restants avant panne — à partir de 21 capteurs de télémétrie, en utilisant deux modèles de régression supervisée comparés rigoureusement.

Le livrable final est un **dashboard de maintenance** qui classe chaque moteur en 3 niveaux d'urgence actionnables : CRITIQUE, ATTENTION, SAIN.

---

## Résultats

### Comparaison des modèles

| Modèle | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest (baseline) | 20.03 cycles | 15.07 cycles | 0.767 |
| **XGBoost (final)** | **19.51 cycles** | **14.38 cycles** | **0.779** |

### Score sur le test set NASA (ground truth officielle)

| Métrique | Valeur |
|---|---|
| RMSE | 28.90 cycles |
| MAE | 20.25 cycles |
| R² | 0.711 |
| NASA Score (PHM 2008) | **10 904** (seuil compétitif < 50 000) |

### Stabilité — Cross-Validation 5-Fold

| Fold | 1 | 2 | 3 | 4 | 5 | Moyenne |
|---|---|---|---|---|---|---|
| RMSE | 19.56 | 19.22 | 19.35 | 19.29 | 19.48 | **19.38 ± 0.12** |

L'écart-type de 0.12 confirme que le modèle est stable et généralisable — il n'a pas mémorisé les données d'entraînement.

---

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) — FD002**

| | Train | Test |
|---|---|---|
| Lignes | 53 759 | 33 991 |
| Moteurs | 260 | 259 |
| Colonnes | 26 | 26 |
| Conditions de vol | 6 | 6 |

FD002 est le sous-dataset le plus réaliste de la suite C-MAPSS, avec 6 conditions de vol simultanées (vs 1 pour FD001). C'est la référence standard dans la littérature pour comparer les algorithmes de pronostic de pannes.

Le dataset n'est pas inclus dans ce repository. Voir `data/README.md` pour les instructions de téléchargement.

**Référence :** Saxena et al., *PHM 2008 — Damage propagation modeling for aircraft engine run-to-failure simulation.*

---

## Structure du projet

```
maintenance-predictive-turbofan/
│
├── README.md                              # Ce fichier
│
├── notebook/
│   └── NASA_finale_v1.ipynb               # Pipeline complet — 10 étapes commentées
│
├── presentation/
│   └── ML_Turbofan_v3.pptx               # Présentation 18 slides avec storytelling  
│
└── data/
    ├── README.md                          # Instructions de téléchargement
    └── .gitkeep
```

---

## Pipeline méthodologique

Le notebook suit 10 étapes numérotées, de la donnée brute à la décision de maintenance.

### Étape 1 — Chargement & Exploration (EDA)

Chargement du dataset FD002 via `kagglehub`. Exploration de la structure, des distributions et de la dégradation des capteurs au fil des cycles.

Points clés découverts :
- Durée de vie des moteurs : 128 à 378 cycles, moyenne 207 cycles
- 2 capteurs quasi-constants identifiés (s_16, s_19)
- Le capteur s_15 montre la tendance de dégradation la plus claire

### Étape 2 — Nettoyage

Pipeline en 5 actions appliquées dans l'ordre, uniquement sur le train puis répliquées sur le test (anti data leakage) :

| Action | Résultat |
|---|---|
| Valeurs manquantes | Aucune dans FD002 |
| Doublons | Aucun |
| Outliers | Clipping 3×IQR — conservateur pour capteurs industriels |
| Capteurs faible variance | s_16, s_19 supprimés (variance < 0.01) |
| Types | Conversion numérique vérifiée |

### Étape 3 — Feature Engineering

Trois décisions cruciales qui font la différence entre un modèle médiocre et un modèle performant :

**Calcul du RUL**
```
RUL = cycle_max (par moteur) − cycle_actuel
```
Chaque ligne reçoit son nombre de cycles restants avant panne.

**RUL Cap à 125 cycles**
```
RUL_final = min(RUL, 125)
```
En début de vie, les capteurs ne montrent aucun signe de dégradation — impossible de distinguer 200 cycles restants de 300. Le plafonnement force le modèle à se concentrer sur la phase de dégradation réelle. Amélioration RMSE estimée à ~15%. Référence : Saxena et al., PHM 2008.

**Rolling Mean — fenêtre 5 cycles**
```
signal_lissé = rolling(window=5, min_periods=1).mean() par moteur
```
Les capteurs industriels sont bruités. La moyenne glissante révèle la tendance de dégradation sous le bruit. Groupby par moteur — aucun leakage entre moteurs.

Résultat : **41 features** (19 capteurs bruts + 19 rolling mean + 3 paramètres opérationnels)

### Étape 4 — Normalisation

```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit sur train uniquement
X_test_scaled  = scaler.transform(X_test)        # transform uniquement sur test
```

Split : 80% train (43 007 lignes) / 20% validation (10 752 lignes)

### Étape 5 — Random Forest (baseline)

```python
rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
```

Rôle : établir un point de référence. Si XGBoost n'améliore pas le Random Forest, c'est qu'il y a un problème dans le pipeline.

### Étape 6 — XGBoost (modèle final)

```python
xgb = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
```

Différence clé avec Random Forest : les arbres sont construits en séquence, chaque arbre corrige les erreurs du précédent (gradient boosting). Plus lent à entraîner, systématiquement plus précis sur les données tabulaires.

### Étape 7 — Évaluation complète

Trois niveaux de validation complémentaires :

1. **Score de validation** (split 80/20) — résultat immédiat pendant le développement
2. **Score test NASA** (ground truth officielle, 259 moteurs) — résultat final réel
3. **NASA Score PHM 2008** — métrique asymétrique officielle : prédire trop tard (manquer une panne) est plus pénalisé que prédire trop tôt

```python
def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1))
```

Score obtenu : **10 904** (seuil < 50 000 = performance compétitive sur FD002)

### Étape 8 — Interprétabilité SHAP

```python
explainer   = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X_sample)
```

Trois visualisations :
- **Bar plot** : importance globale — s_15 domine avec un impact moyen de 11.4
- **Beeswarm** : direction d'influence — valeurs élevées de s_15 réduisent le RUL prédit
- **Waterfall** : décomposition d'un moteur critique — pourquoi RUL = 0.8 cycles pour le moteur N°102

### Étape 9 — Dashboard de maintenance

Classification des 259 moteurs du test set :

| Niveau | Seuil RUL | Moteurs | Action |
|---|---|---|---|
| CRITIQUE | RUL ≤ 30 cycles | ~38 (15%) | Arrêt immédiat — inspection d'urgence |
| ATTENTION | 30 < RUL ≤ 80 cycles | ~65 (25%) | Maintenance sous 5 vols |
| SAIN | RUL > 80 cycles | ~156 (60%) | Surveillance normale |

---

## Installation

### Prérequis

```
Python >= 3.9
```

### Dépendances

```bash
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn kagglehub
```

### Lancer le notebook

```bash
jupyter notebook notebook/NASA_finale_v1.ipynb
```

Le dataset est téléchargé automatiquement à la cellule 4 via `kagglehub`. Une authentification Kaggle est nécessaire (voir `data/README.md`).

---

## Axes d'amélioration identifiés

**Court terme**
- Intégrer le dashboard dans un système de monitoring de flotte
- Alertes automatiques pour les moteurs CRITIQUE

**Moyen terme**
- Normalisation par moteur (`MinMaxScaler` per `unit_nr`) — limite actuelle : le scaler global est influencé par les conditions de vol
- Tester des fenêtres rolling de 10 et 15 cycles
- Valider sur FD001, FD003, FD004

**Long terme**
- Architectures LSTM ou Transformer pour capturer les dépendances temporelles longues
- Déploiement API FastAPI + Docker
- Pipeline de réentraînement automatique sur données terrain

---

## Stack technique

| Catégorie | Outils |
|---|---|
| Machine Learning | XGBoost, Scikit-learn RandomForestRegressor |
| Feature Engineering | Pandas rolling, NumPy log1p, MinMaxScaler |
| Évaluation | RMSE, MAE, R², KFold cross-validation, NASA Score PHM 2008 |
| Interprétabilité | SHAP TreeExplainer |
| Visualisation | Matplotlib, Seaborn |
| Data | Pandas, NumPy, kagglehub |

---

## Auteur

Projet réalisé dans le cadre du **DU Data Analytics**
Université Paris 1 Panthéon-Sorbonne
Cours : Machine Learning
