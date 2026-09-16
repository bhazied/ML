# 🤖 Deep Q-Learning pour l'Organisation d'Entrepot

Un prototype complet d'apprentissage par renforcement pour optimiser la logistique d'entrepot en utilisant le **Deep Q-Learning (DQN)**.

## 📋 Table des matieres

- [Apercu](#apercu)
- [Fonctionnalites](#fonctionnalites)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Resultats](#resultats)
- [Structure du projet](#structure-du-projet)
- [Contribuer](#contribuer)
- [License](#license)

## 🎯 Apercu

Ce projet implemente un agent intelligent capable d'apprendre a organiser un entrepot de maniere autonome en minimisant les deplacements et en maximisant les livraisons.

**Probleme** : Un robot doit recuperer des articles disperses dans un entrepot et les deposer a une zone fixe, tout en evitant les obstacles.

**Solution** : Modelling du probleme comme un **Processus de Decision Markovien (MDP)** et resolution avec **Deep Q-Network (DQN)**.

## ✨ Fonctionnalites

- ✅ Environnement simule d'entrepot (grille 10x10)
- ✅ Agent DQN avec replay buffer et target network
- ✅ Visualisation des performances en temps reel
- ✅ Code modulaire et extensible
- ✅ Compatible avec Gym/Gymnasium
- ✅ Version TensorFlow et NumPy

## 📦 Installation

### Pre-requis

- Python 3.8+
- pip

### Dependencies

```bash
pip install numpy matplotlib tensorflow gymnasium
```

### Cloner le depot

```bash
git clone https://github.com/bhazied/ML.git
cd ML/dqn_entropot
```

## 🚀 Utilisation

### Lancer l'entrainement

```bash
python train_dqn.py
```

### Options d'entrainement

```python
# Modifier les hyperparametres dans train_dqn.py
episodes = 500  # Nombre d'episodes
grid_size = 10  # Taille de la grille
n_items = 5     # Nombre d'articles
```

### Visualiser les resultats

Les graphiques sont sauvegardes dans le dossier `output/` :
- `dqn_training_results.png` : Progression des recompenses
- `warehouse_grid.png` : Grille de l'entrepot
- `mdp_diagram.png` : Diagramme MDP
- `dqn_architecture.png` : Architecture du reseau

## 🏗️ Architecture

### Processus de Decision Markovien (MDP)

| Composant | Description |
|-----------|-------------|
| **Etats (S)** | Position du robot + etat des articles (13 features) |
| **Actions (A)** | 6 actions : Haut, Bas, Gauche, Droite, Saisir, Deposer |
| **Transitions (T)** | Deterministe avec penalites pour obstacles |
| **Recompenses (R)** | -1 (mouvement), +5 (saisir), +10 (deposer), -10 (collision), +20 (bonus) |

### Deep Q-Network (DQN)

```
Entree (13) -> Dense(64, ReLU) -> Dense(64, ReLU) -> Sortie(6)
```

**Hyperparametres** :
- Learning rate : 0.001
- Gamma (discount) : 0.95
- Epsilon start : 1.0
- Epsilon min : 0.01
- Epsilon decay : 0.995
- Replay buffer : 2000 experiences
- Batch size : 32

## 📊 Resultats

Apres 300 episodes d'entrainement :

- **Recompense moyenne** : -200 (convergence vers politique optimale)
- **Articles livres** : Amelioration progressive avec l'exploration
- **Epsilon** : Decay de 1.0 a 0.01 (exploitation croissante)

![Training Results](output/dqn_training_results.png)

## 📁 Structure du projet

```
dqn-warehouse/
├── README.md
├── train_dqn.py              # Script principal d'entrainement
├── warehouse_env.py          # Environnement simule
├── dqn_agent.py              # Agent DQN
├── dqn_network_tf.py         # Version TensorFlow
├── requirements.txt          # Dependencies
├── output/                   # Visualisations
│   ├── dqn_training_results.png
│   ├── warehouse_grid.png
│   ├── mdp_diagram.png
│   └── dqn_architecture.png
└── notebooks/
    └── tutorial.ipynb        # Notebook interactif
```

## 🧪 Tests

```bash
python -m pytest tests/
```

## 🤝 Contribuer

1. Fork le projet
2. Creer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Distribue sous la licence MIT. Voir `LICENSE` pour plus d'informations.

## 👤 Auteurs

- **Zied Ben Hadj Amor** - *Travail initial* - [bhazied](https://github.com/bhazied)

## 🙏 Remerciements

- OpenAI Gym pour l'inspiration de l'environnement
- DeepMind pour l'algorithme DQN original
- La communaute RL pour les ressources et tutoriels

## 📚 References

1. Mnih, V. et al. "Human-level control through deep reinforcement learning." Nature, 2015.
2. Sutton, R.S. & Barto, A.G. "Reinforcement Learning: An Introduction." MIT Press, 2018.
3. [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course)

---

**Projet realise dans le cadre d'un article LinkedIn sur l'apprentissage par renforcement applique a la logistique.**

#ReinforcementLearning #DeepLearning #MDP #Qlearning #IA #Robotique #Entrepot #Innovation #Tech #DataScience #AI #MachineLearning #Automation
