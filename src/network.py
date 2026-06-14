from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def batch_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mu) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mu, var, gamma, beta, eps)
    return out, cache


def batch_norm_backward(dout: np.ndarray, cache):
    x, x_norm, mu, var, gamma, beta, eps = cache
    N, D = x.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mu) * -0.5 * (var + eps) ** (-1.5), axis=0)
    dmu = np.sum(dx_norm * -1.0 / np.sqrt(var + eps), axis=0) + dvar * np.sum(-2.0 * (x - mu), axis=0) / N
    dx = dx_norm / np.sqrt(var + eps) + dvar * 2.0 * (x - mu) / N + dmu / N
    return dx, dgamma, dbeta


@dataclass
class NetworkConfig:
    input_size: int = 784
    hidden_size: int = 64
    output_size: int = 10
    learning_rate: float = 0.1
    batch_size: int = 128
    seed: int = 42
    weight_decay: float = 0.2


class SimpleMLP:
    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        h1 = config.hidden_size * 2
        h2 = config.hidden_size + 32
        h3 = config.hidden_size + 32
        h4 = config.hidden_size

        self.params: dict[str, np.ndarray] = {
            "W1": (
                rng.standard_normal((config.input_size, h1))
                * np.sqrt(2.0 / config.input_size)
            ).astype(np.float32),
            "b1": np.zeros(h1, dtype=np.float32),
            "W2": (
                rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1)
            ).astype(np.float32),
            "b2": np.zeros(h2, dtype=np.float32),
            "W3": (
                rng.standard_normal((h2, h3)) * np.sqrt(2.0 / h2)
            ).astype(np.float32),
            "b3": np.zeros(h3, dtype=np.float32),
            "W4": (
                rng.standard_normal((h3, h4)) * np.sqrt(2.0 / h3)
            ).astype(np.float32),
            "b4": np.zeros(h4, dtype=np.float32),
            "W5": (
                rng.standard_normal((h4, config.output_size)) * np.sqrt(2.0 / h4)
            ).astype(np.float32),
            "b5": np.zeros(config.output_size, dtype=np.float32),
            "gamma1": np.ones(h1, dtype=np.float32),
            "beta1": np.zeros(h1, dtype=np.float32),
            "gamma2": np.ones(h2, dtype=np.float32),
            "beta2": np.zeros(h2, dtype=np.float32),
            "gamma3": np.ones(h3, dtype=np.float32),
            "beta3": np.zeros(h3, dtype=np.float32),
            "gamma4": np.ones(h4, dtype=np.float32),
            "beta4": np.zeros(h4, dtype=np.float32),
        }
        self.accumulators: dict[str, np.ndarray] = {
            key: np.zeros_like(value, dtype=np.float32)
            for key, value in self.params.items()
        }

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z1 = _relu(np.dot(x, self.params["W1"]) + self.params["b1"])
        z1, _ = batch_norm_forward(z1, self.params["gamma1"], self.params["beta1"]) 
        z2 = _relu(np.dot(z1, self.params["W2"]) + self.params["b2"])
        z2, _ = batch_norm_forward(z2, self.params["gamma2"], self.params["beta2"]) 
        z3 = _relu(np.dot(z2, self.params["W3"]) + self.params["b3"])
        z3, _ = batch_norm_forward(z3, self.params["gamma3"], self.params["beta3"]) 
        z4 = _relu(np.dot(z3, self.params["W4"]) + self.params["b4"])
        z4, _ = batch_norm_forward(z4, self.params["gamma4"], self.params["beta4"]) 
        logits = np.dot(z4, self.params["W5"]) + self.params["b5"]
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        correct = 0
        total = x.shape[0]
        batch_size = self.config.batch_size
        for i in range(0, total, batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            pred = self.predict(x_batch)
            correct += int(np.sum(pred == y_batch))
        return float(correct) / float(total)

    def train_epoch(self, x: np.ndarray, y: np.ndarray, epoch: int) -> float:
        rng = np.random.default_rng(self.config.seed + epoch)
        indices = rng.permutation(x.shape[0])
        total_loss = 0.0
        steps = 0
        batch_size = self.config.batch_size
        i = 1
        for start in range(0, x.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            z1_linear = np.dot(x_batch, self.params["W1"]) + self.params["b1"]
            z1 = _relu(z1_linear)
            z1, cache1 = batch_norm_forward(z1, self.params["gamma1"], self.params["beta1"]) 
            z2_linear = np.dot(z1, self.params["W2"]) + self.params["b2"]
            z2 = _relu(z2_linear)
            z2, cache2 = batch_norm_forward(z2, self.params["gamma2"], self.params["beta2"]) 
            z3_linear = np.dot(z2, self.params["W3"]) + self.params["b3"]
            z3 = _relu(z3_linear)
            z3, cache3 = batch_norm_forward(z3, self.params["gamma3"], self.params["beta3"]) 
            z4_linear = np.dot(z3, self.params["W4"]) + self.params["b4"]
            z4 = _relu(z4_linear)
            z4, cache4 = batch_norm_forward(z4, self.params["gamma4"], self.params["beta4"]) 
            logits = np.dot(z4, self.params["W5"]) + self.params["b5"]
            probs = _softmax(logits)

            y_one_hot = _one_hot(y_batch, self.config.output_size)
            loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
            total_loss += float(loss)
            steps += 1

            d_logits = (probs - y_one_hot) / x_batch.shape[0]
            dW5 = np.dot(z4.T, d_logits)
            db5 = np.sum(d_logits, axis=0)

            d_z4_bn = np.dot(d_logits, self.params["W5"].T)
            d_z4, dgamma4, dbeta4 = batch_norm_backward(d_z4_bn, cache4)
            d_z4_linear = d_z4 * (z4_linear > 0).astype(np.float32)
            dW4 = np.dot(z3.T, d_z4_linear)
            db4 = np.sum(d_z4_linear, axis=0)

            d_z3_bn = np.dot(d_z4_linear, self.params["W4"].T)
            d_z3, dgamma3, dbeta3 = batch_norm_backward(d_z3_bn, cache3)
            d_z3_linear = d_z3 * (z3_linear > 0).astype(np.float32)
            dW3 = np.dot(z2.T, d_z3_linear)
            db3 = np.sum(d_z3_linear, axis=0)

            d_z2_bn = np.dot(d_z3_linear, self.params["W3"].T)
            d_z2, dgamma2, dbeta2 = batch_norm_backward(d_z2_bn, cache2)
            d_z2_linear = d_z2 * (z2_linear > 0).astype(np.float32)
            dW2 = np.dot(z1.T, d_z2_linear)
            db2 = np.sum(d_z2_linear, axis=0)

            d_z1_bn = np.dot(d_z2_linear, self.params["W2"].T)
            d_z1, dgamma1, dbeta1 = batch_norm_backward(d_z1_bn, cache1)
            d_z1_linear = d_z1 * (z1_linear > 0).astype(np.float32)
            dW1 = np.dot(x_batch.T, d_z1_linear)
            db1 = np.sum(d_z1_linear, axis=0)

           

            lr = self.config.learning_rate
            weight_decay = self.config.weight_decay
            epsilon = 1e-8
            dW1 = dW1 + weight_decay * self.params["W1"]
            dW2 = dW2 + weight_decay * self.params["W2"]
            dW3 = dW3 + weight_decay * self.params["W3"]
            dW4 = dW4 + weight_decay * self.params["W4"]
            dW5 = dW5 + weight_decay * self.params["W5"]

            # Accumulate squared gradients (Adagrad-like)
            self.accumulators["W1"] += dW1 ** 2
            self.accumulators["W2"] += dW2 ** 2
            self.accumulators["W3"] += dW3 ** 2
            self.accumulators["W4"] += dW4 ** 2
            self.accumulators["W5"] += dW5 ** 2
            self.accumulators["b1"] += db1 ** 2
            self.accumulators["b2"] += db2 ** 2
            self.accumulators["b3"] += db3 ** 2
            self.accumulators["b4"] += db4 ** 2
            self.accumulators["b5"] += db5 ** 2
            self.accumulators["gamma1"] += dgamma1 ** 2
            self.accumulators["beta1"] += dbeta1 ** 2
            self.accumulators["gamma2"] += dgamma2 ** 2
            self.accumulators["beta2"] += dbeta2 ** 2
            self.accumulators["gamma3"] += dgamma3 ** 2
            self.accumulators["beta3"] += dbeta3 ** 2
            self.accumulators["gamma4"] += dgamma4 ** 2
            self.accumulators["beta4"] += dbeta4 ** 2

            # Parameter updates
            self.params["W1"] -= (lr / np.sqrt(self.accumulators["W1"] + epsilon)) * dW1.astype(np.float32)
            self.params["b1"] -= (lr / np.sqrt(self.accumulators["b1"] + epsilon)) * db1.astype(np.float32)
            self.params["W2"] -= (lr / np.sqrt(self.accumulators["W2"] + epsilon)) * dW2.astype(np.float32)
            self.params["b2"] -= (lr / np.sqrt(self.accumulators["b2"] + epsilon)) * db2.astype(np.float32)
            self.params["W3"] -= (lr / np.sqrt(self.accumulators["W3"] + epsilon)) * dW3.astype(np.float32)
            self.params["b3"] -= (lr / np.sqrt(self.accumulators["b3"] + epsilon)) * db3.astype(np.float32)
            self.params["W4"] -= (lr / np.sqrt(self.accumulators["W4"] + epsilon)) * dW4.astype(np.float32)
            self.params["b4"] -= (lr / np.sqrt(self.accumulators["b4"] + epsilon)) * db4.astype(np.float32)
            self.params["W5"] -= (lr / np.sqrt(self.accumulators["W5"] + epsilon)) * dW5.astype(np.float32)
            self.params["b5"] -= (lr / np.sqrt(self.accumulators["b5"] + epsilon)) * db5.astype(np.float32)
            self.params["gamma1"] -= (lr / np.sqrt(self.accumulators["gamma1"] + epsilon)) * dgamma1.astype(np.float32)
            self.params["beta1"] -= (lr / np.sqrt(self.accumulators["beta1"] + epsilon)) * dbeta1.astype(np.float32)
            self.params["gamma2"] -= (lr / np.sqrt(self.accumulators["gamma2"] + epsilon)) * dgamma2.astype(np.float32)
            self.params["beta2"] -= (lr / np.sqrt(self.accumulators["beta2"] + epsilon)) * dbeta2.astype(np.float32)
            self.params["gamma3"] -= (lr / np.sqrt(self.accumulators["gamma3"] + epsilon)) * dgamma3.astype(np.float32)
            self.params["beta3"] -= (lr / np.sqrt(self.accumulators["beta3"] + epsilon)) * dbeta3.astype(np.float32)
            self.params["gamma4"] -= (lr / np.sqrt(self.accumulators["gamma4"] + epsilon)) * dgamma4.astype(np.float32)
            self.params["beta4"] -= (lr / np.sqrt(self.accumulators["beta4"] + epsilon)) * dbeta4.astype(np.float32)

        return total_loss / max(steps, 1)

    def to_state(self) -> dict[str, object]:
        return {
            "model_type": "SimpleMLP",
            "config": {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "output_size": self.config.output_size,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
                "weight_decay": self.config.weight_decay,
            },
            "params": self.params,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleMLP":
        config_obj = state.get("config")
        if not isinstance(config_obj, dict):
            raise ValueError("Invalid state: 'config' must be a dict")
        config_dict: dict[str, Any] = config_obj

        config = NetworkConfig(
            input_size=int(config_dict["input_size"]),
            hidden_size=int(config_dict["hidden_size"]),
            output_size=int(config_dict["output_size"]),
            learning_rate=float(config_dict.get("learning_rate", 0.1)),
            batch_size=int(config_dict.get("batch_size", 128)),
            seed=int(config_dict.get("seed", 42)),
            weight_decay=float(config_dict.get("weight_decay", 0.1)),
        )

        params_obj = state.get("params")
        if not isinstance(params_obj, dict):
            raise ValueError("Invalid state: 'params' must be a dict")
        params: dict[str, np.ndarray] = {}
        for key, value in params_obj.items():
            if not isinstance(key, str) or not isinstance(value, np.ndarray):
                raise ValueError("Invalid state: params must be dict[str, np.ndarray]")
            params[key] = value

        model = cls(config)
        model.params = params
        return model
        model = cls(config)
        model.params = params
        return model
