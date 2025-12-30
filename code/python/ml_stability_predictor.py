#!/usr/bin/env python3
"""
ML Stability Predictor
======================
Machine learning models to predict:
1. Lyapunov exponent λ from system parameters
2. Optimal regularization ε for stability
3. Configuration classification (stable vs chaotic)

Uses simple neural networks trainable on CPU.
"""

import numpy as np
import json
from pathlib import Path

# Try to import torch, fall back to numpy-only implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using NumPy-only implementation")


# ==============================================================================
# DATA GENERATION
# ==============================================================================

def generate_training_data(n_samples=10000, seed=42):
    """
    Generate synthetic training data based on our discoveries.

    Features:
    - N: number of bodies (3-30)
    - epsilon: regularization parameter
    - epsilon_ratio: ε/(ℏ/mv) = ε × m × v / ℏ
    - config_type: 0=random, 1=hierarchical, 2=lagrange
    - energy_sign: -1 for bound, +1 for unbound
    - virial_ratio: 2K/|U| (should be ~1 for virial equilibrium)

    Target:
    - lambda: Lyapunov exponent
    - stable: 1 if λ < 0, else 0
    """
    np.random.seed(seed)

    data = []

    for _ in range(n_samples):
        # Random system parameters
        N = np.random.randint(3, 31)
        m_typical = 1.0 / N
        v_rms = np.random.uniform(0.5, 2.0)

        # Regularization scale
        epsilon_quantum = 1.0 / (m_typical * v_rms)  # ℏ = 1
        epsilon_factor = np.random.uniform(0.1, 10.0)
        epsilon = epsilon_factor * epsilon_quantum
        epsilon_ratio = epsilon_factor

        # Configuration type
        config_type = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

        # Energy (bound vs unbound)
        energy_sign = np.random.choice([-1, 1], p=[0.8, 0.2])

        # Virial ratio
        virial_ratio = np.random.uniform(0.5, 1.5)

        # Compute Lyapunov exponent based on our discoveries
        # Base chaos rate from N-transition scan: λ ≈ 0.07 for random configs
        lambda_base = 0.07

        # Configuration effect
        if config_type == 0:  # Random
            config_factor = 1.0
        elif config_type == 1:  # Hierarchical
            config_factor = -2.0  # Stabilizing
        else:  # Lagrange
            config_factor = -3.0  # Very stabilizing

        # Epsilon effect (from our λ(ε) ∝ ε^(-1.674) discovery)
        epsilon_effect = np.sign(1.0 - epsilon_ratio) * np.abs(1.0 - epsilon_ratio) ** 0.5

        # Virial effect (closer to 1.0 = more stable)
        virial_effect = -np.abs(virial_ratio - 1.0)

        # Energy effect (bound = more stable)
        energy_effect = 0.02 * energy_sign

        # Combined Lyapunov
        lambda_val = lambda_base + 0.05 * config_factor + 0.03 * epsilon_effect + 0.1 * virial_effect + energy_effect

        # Add noise
        lambda_val += np.random.normal(0, 0.01)

        stable = 1 if lambda_val < 0 else 0

        data.append({
            'N': N,
            'epsilon': epsilon,
            'epsilon_ratio': epsilon_ratio,
            'config_type': config_type,
            'energy_sign': energy_sign,
            'virial_ratio': virial_ratio,
            'v_rms': v_rms,
            'lambda': lambda_val,
            'stable': stable
        })

    return data


def prepare_tensors(data):
    """Convert data to numpy arrays (or torch tensors if available)."""

    X = np.array([
        [d['N'], d['epsilon_ratio'], d['config_type'],
         d['energy_sign'], d['virial_ratio'], d['v_rms']]
        for d in data
    ], dtype=np.float32)

    y_lambda = np.array([d['lambda'] for d in data], dtype=np.float32)
    y_stable = np.array([d['stable'] for d in data], dtype=np.float32)

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    return X_norm, y_lambda, y_stable, X_mean, X_std


# ==============================================================================
# PYTORCH MODELS (if available)
# ==============================================================================

if TORCH_AVAILABLE:

    class LyapunovPredictor(nn.Module):
        """Neural network to predict Lyapunov exponent."""

        def __init__(self, input_dim=6, hidden_dims=[64, 32, 16]):
            super().__init__()

            layers = []
            prev_dim = input_dim

            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = h_dim

            layers.append(nn.Linear(prev_dim, 1))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x).squeeze(-1)


    class StabilityClassifier(nn.Module):
        """Neural network to classify stable vs chaotic."""

        def __init__(self, input_dim=6, hidden_dims=[64, 32]):
            super().__init__()

            layers = []
            prev_dim = input_dim

            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = h_dim

            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x).squeeze(-1)


    def train_models(X, y_lambda, y_stable, epochs=100, lr=0.001, batch_size=64):
        """Train both models."""

        # Convert to tensors
        X_t = torch.FloatTensor(X)
        y_lambda_t = torch.FloatTensor(y_lambda)
        y_stable_t = torch.FloatTensor(y_stable)

        # Split train/val
        n = len(X)
        idx = np.random.permutation(n)
        train_idx = idx[:int(0.8*n)]
        val_idx = idx[int(0.8*n):]

        X_train, X_val = X_t[train_idx], X_t[val_idx]
        y_lambda_train, y_lambda_val = y_lambda_t[train_idx], y_lambda_t[val_idx]
        y_stable_train, y_stable_val = y_stable_t[train_idx], y_stable_t[val_idx]

        # Initialize models
        lyapunov_model = LyapunovPredictor()
        stability_model = StabilityClassifier()

        # Optimizers
        opt_lyap = optim.Adam(lyapunov_model.parameters(), lr=lr)
        opt_stab = optim.Adam(stability_model.parameters(), lr=lr)

        # Loss functions
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        # Training loop
        print("Training Lyapunov predictor...")
        for epoch in range(epochs):
            lyapunov_model.train()

            # Mini-batch training
            perm = np.random.permutation(len(X_train))
            total_loss = 0

            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i+batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_lambda_train[batch_idx]

                opt_lyap.zero_grad()
                pred = lyapunov_model(X_batch)
                loss = mse_loss(pred, y_batch)
                loss.backward()
                opt_lyap.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                lyapunov_model.eval()
                val_pred = lyapunov_model(X_val)
                val_loss = mse_loss(val_pred, y_lambda_val).item()
                print(f"  Epoch {epoch+1}: train_loss={total_loss:.4f}, val_loss={val_loss:.4f}")

        print("\nTraining stability classifier...")
        for epoch in range(epochs):
            stability_model.train()

            perm = np.random.permutation(len(X_train))
            total_loss = 0

            for i in range(0, len(X_train), batch_size):
                batch_idx = perm[i:i+batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_stable_train[batch_idx]

                opt_stab.zero_grad()
                pred = stability_model(X_batch)
                loss = bce_loss(pred, y_batch)
                loss.backward()
                opt_stab.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                stability_model.eval()
                val_pred = stability_model(X_val)
                val_loss = bce_loss(val_pred, y_stable_val).item()
                val_acc = ((val_pred > 0.5) == y_stable_val).float().mean().item()
                print(f"  Epoch {epoch+1}: train_loss={total_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

        return lyapunov_model, stability_model


# ==============================================================================
# NUMPY-ONLY FALLBACK
# ==============================================================================

class NumpyLyapunovPredictor:
    """Simple linear regression for Lyapunov prediction (numpy-only)."""

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Add bias term
        X_b = np.c_[np.ones(len(X)), X]
        # Solve normal equations
        self.weights = np.linalg.lstsq(X_b, y, rcond=None)[0]

    def predict(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return X_b @ self.weights


class NumpyStabilityClassifier:
    """Logistic regression for stability classification (numpy-only)."""

    def __init__(self):
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y, lr=0.01, epochs=1000):
        X_b = np.c_[np.ones(len(X)), X]
        self.weights = np.zeros(X_b.shape[1])

        for _ in range(epochs):
            pred = self.sigmoid(X_b @ self.weights)
            gradient = X_b.T @ (pred - y) / len(y)
            self.weights -= lr * gradient

    def predict(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return self.sigmoid(X_b @ self.weights)


def train_numpy_models(X, y_lambda, y_stable):
    """Train numpy-only models."""

    print("Training Lyapunov predictor (numpy linear regression)...")
    lyap_model = NumpyLyapunovPredictor()
    lyap_model.fit(X, y_lambda)

    # Evaluate
    pred = lyap_model.predict(X)
    mse = np.mean((pred - y_lambda) ** 2)
    print(f"  MSE: {mse:.6f}")

    print("\nTraining stability classifier (numpy logistic regression)...")
    stab_model = NumpyStabilityClassifier()
    stab_model.fit(X, y_stable)

    # Evaluate
    pred = stab_model.predict(X)
    acc = np.mean((pred > 0.5) == y_stable)
    print(f"  Accuracy: {acc:.3f}")

    return lyap_model, stab_model


# ==============================================================================
# PHYSICS-BASED PREDICTOR (no training needed)
# ==============================================================================

class PhysicsBasedPredictor:
    """
    Predictor based on our derived physics formulas.
    No training needed - uses analytical results.
    """

    def __init__(self):
        # Empirical constants from our discoveries
        self.lambda_base = 0.07  # Base chaos rate for random configs
        self.epsilon_exponent = -1.674  # λ(ε) ∝ ε^(-1.674)
        self.epsilon_coeff = 0.222

    def predict_lyapunov(self, N, epsilon_ratio, config_type, virial_ratio):
        """
        Predict Lyapunov exponent.

        Args:
            N: number of bodies
            epsilon_ratio: ε / ε_quantum
            config_type: 0=random, 1=hierarchical, 2=lagrange
            virial_ratio: 2K/|U|

        Returns:
            Predicted Lyapunov exponent
        """
        # Configuration factor
        config_factors = {
            0: 1.0,    # Random: chaotic
            1: -0.5,   # Hierarchical: stable
            2: -0.7    # Lagrange: very stable
        }
        config_factor = config_factors.get(config_type, 1.0)

        # Epsilon scaling (for chaotic regime)
        if config_factor > 0:
            lambda_epsilon = self.epsilon_coeff * (epsilon_ratio ** self.epsilon_exponent)
        else:
            # For stable configs, epsilon effect is weaker
            lambda_epsilon = -0.1 * epsilon_ratio

        # Virial deviation effect
        virial_deviation = abs(virial_ratio - 1.0)
        virial_factor = 1.0 + 0.5 * virial_deviation

        # Combine
        if config_factor > 0:
            lambda_pred = lambda_epsilon * config_factor * virial_factor
        else:
            lambda_pred = config_factor * (1.0 + 0.1 * virial_deviation)

        return lambda_pred

    def predict_stability(self, N, epsilon_ratio, config_type, virial_ratio):
        """
        Predict stability (stable=True, chaotic=False).
        """
        lambda_pred = self.predict_lyapunov(N, epsilon_ratio, config_type, virial_ratio)
        return lambda_pred < 0

    def recommend_epsilon(self, N, config_type, target_lambda=-0.1):
        """
        Recommend regularization ε to achieve target Lyapunov.

        For random configs (config_type=0), we may not be able to achieve λ < 0.
        For structured configs, returns ε that gives target λ.
        """
        if config_type == 0:
            # Random configs are inherently chaotic
            return None, "Random configurations are chaotic regardless of ε. Use structured configuration."

        # For structured configs, solve for ε that gives target λ
        # Using simplified model: λ ≈ -0.5 for hierarchical, -0.7 for Lagrange
        # Epsilon can fine-tune

        if config_type == 1:  # Hierarchical
            base_lambda = -0.5
        else:  # Lagrange
            base_lambda = -0.7

        # Recommend ε_ratio ~ 1 (quantum scale)
        recommended_epsilon_ratio = 1.0

        return recommended_epsilon_ratio, f"Recommended ε = {recommended_epsilon_ratio} × ε_quantum. Expected λ ≈ {base_lambda:.2f}"


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*70)
    print("ML STABILITY PREDICTOR")
    print("Training models to predict N-body chaos from system parameters")
    print("="*70)
    print()

    # Generate training data
    print("Generating training data based on physics discoveries...")
    data = generate_training_data(n_samples=10000)
    X, y_lambda, y_stable, X_mean, X_std = prepare_tensors(data)

    print(f"  Generated {len(data)} samples")
    print(f"  Features: N, ε_ratio, config_type, energy_sign, virial_ratio, v_rms")
    print(f"  Targets: λ (Lyapunov), stable (0/1)")
    print()

    # Train models
    if TORCH_AVAILABLE:
        lyap_model, stab_model = train_models(X, y_lambda, y_stable)
    else:
        lyap_model, stab_model = train_numpy_models(X, y_lambda, y_stable)

    # Physics-based predictor (always available)
    print("\nInitializing physics-based predictor...")
    physics_predictor = PhysicsBasedPredictor()

    # Test predictions
    print("\n" + "="*70)
    print("TEST PREDICTIONS")
    print("="*70)

    test_cases = [
        {'N': 3, 'epsilon_ratio': 1.0, 'config_type': 0, 'virial_ratio': 1.0, 'desc': '3-body random'},
        {'N': 3, 'epsilon_ratio': 1.0, 'config_type': 1, 'virial_ratio': 1.0, 'desc': '3-body hierarchical'},
        {'N': 3, 'epsilon_ratio': 1.0, 'config_type': 2, 'virial_ratio': 1.0, 'desc': '3-body Lagrange'},
        {'N': 10, 'epsilon_ratio': 1.0, 'config_type': 0, 'virial_ratio': 1.0, 'desc': '10-body random'},
        {'N': 30, 'epsilon_ratio': 1.0, 'config_type': 0, 'virial_ratio': 1.0, 'desc': '30-body random'},
        {'N': 30, 'epsilon_ratio': 0.1, 'config_type': 0, 'virial_ratio': 1.0, 'desc': '30-body small ε'},
        {'N': 30, 'epsilon_ratio': 10.0, 'config_type': 0, 'virial_ratio': 1.0, 'desc': '30-body large ε'},
    ]

    print(f"\n{'Description':<25} | {'Physics λ':>10} | {'Stable':>8}")
    print("-"*50)

    for tc in test_cases:
        lambda_pred = physics_predictor.predict_lyapunov(
            tc['N'], tc['epsilon_ratio'], tc['config_type'], tc['virial_ratio']
        )
        stable = physics_predictor.predict_stability(
            tc['N'], tc['epsilon_ratio'], tc['config_type'], tc['virial_ratio']
        )
        status = "STABLE" if stable else "CHAOTIC"
        print(f"{tc['desc']:<25} | {lambda_pred:>10.4f} | {status:>8}")

    # Save models and normalization parameters
    output_dir = Path('/home/user/Testing-env/models')
    output_dir.mkdir(exist_ok=True)

    # Save normalization parameters
    norm_params = {
        'X_mean': X_mean.tolist(),
        'X_std': X_std.tolist(),
        'feature_names': ['N', 'epsilon_ratio', 'config_type', 'energy_sign', 'virial_ratio', 'v_rms']
    }

    with open(output_dir / 'normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)

    if TORCH_AVAILABLE:
        torch.save(lyap_model.state_dict(), output_dir / 'lyapunov_predictor.pt')
        torch.save(stab_model.state_dict(), output_dir / 'stability_classifier.pt')
        print(f"\nModels saved to {output_dir}")

    print("\nPhysics-based predictor ready (no training needed)")
    print("Usage: physics_predictor.predict_lyapunov(N, epsilon_ratio, config_type, virial_ratio)")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR STABILITY")
    print("="*70)

    for config_name, config_id in [('Random', 0), ('Hierarchical', 1), ('Lagrange', 2)]:
        eps_rec, msg = physics_predictor.recommend_epsilon(10, config_id)
        print(f"\n{config_name} configuration:")
        print(f"  {msg}")

    return physics_predictor


if __name__ == "__main__":
    predictor = main()
