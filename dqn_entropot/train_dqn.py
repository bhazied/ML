"""
Script principal d'entrainement du DQN pour l'entrepot.
Utilise Gymnasium et le DQN TensorFlow.
"""

import matplotlib.pyplot as plt
import numpy as np

from warehouse_env import WarehouseEnv
from dqn_agent_tf import DQNAgent, train


def main():
    # Création de l'environnement
    env = WarehouseEnv(grid_size=10, n_items=5)

    # Dimensions
    state_size = 2 + 1 + (env.n_items * 2)  # x, y, carrying, items
    action_size = env.action_space

    # Création de l'agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        memory_size=50_000,
        batch_size=64,
    )

    # Entraînement
    print("Démarrage de l'entraînement...")
    history = train(env, agent, episodes=500, max_steps=250, target_update_every=20)

    # Sauvegarde du modèle
    agent.save("models/dqn_warehouse.keras")
    print("Modèle sauvegardé dans models/dqn_warehouse.keras")

    # Visualisation des ré
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history["rewards"], alpha=0.4)
    plt.plot(np.convolve(history["rewards"], np.ones(50) / 50, mode="valid"), linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Récompense totale")
    plt.title("Progression des récompenses")

    plt.subplot(1, 3, 2)
    plt.plot(history["delivered"], alpha=0.4)
    plt.plot(np.convolve(history["delivered"], np.ones(50) / 50, mode="valid"), linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Articles livrés")
    plt.title("Progression des livraisons")

    plt.subplot(1, 3, 3)
    plt.plot(history["losses"], alpha=0.4)
    plt.plot(np.convolve(np.nan_to_num(history["losses"]), np.ones(50) / 50, mode="valid"), linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Perte (loss)")
    plt.title("Perte durant l'entraînement")

    plt.tight_layout()
    plt.savefig("output/training_results.png", dpi=150)
    plt.show()

    print("Graphique sauvegardé dans output/training_results.png")


if __name__ == "__main__":
    main()
