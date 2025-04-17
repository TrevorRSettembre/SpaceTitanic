# Author: Trevor Settembre
# Project Title: SpaceShip Titanic
# Description: This file allows the model to use early stopping to stop any unproductive training


import torch

class MultiMetricEarlyStopping:
    def __init__(self, 
                 monitor_metrics,
                 mode="min",  # or dict per metric
                 patience=5, 
                 delta=0.0, 
                 verbose=False, 
                 path=None, 
                 stop_mode="any"  # "any" or "all"
    ):
        """
        Args:
            monitor_metrics (list): List of metric names to track.
            mode (str or dict): "min" or "max" for each metric (str or dict per metric).
            patience (int): Epochs to wait after no improvement.
            delta (float): Minimum improvement to count as better.
            verbose (bool): Whether to print messages.
            path (str): Optional path to save best model weights.
            stop_mode (str): "any" or "all" — stop if any/all metrics stop improving.
        """
        self.monitor_metrics = monitor_metrics
        self.modes = {m: mode if isinstance(mode, str) else mode.get(m, "min") for m in monitor_metrics}
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.stop_mode = stop_mode

        self.best_metrics = {m: float("inf") if self.modes[m] == "min" else -float("inf") for m in monitor_metrics}
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None

    def _is_improved(self, metric, current):
        mode = self.modes[metric]
        best = self.best_metrics[metric]
        if mode == "min":
            return current < best - self.delta
        elif mode == "max":
            return current > best + self.delta
        else:
            raise ValueError(f"Invalid mode '{mode}' for metric '{metric}'.")

    def __call__(self, current_metrics, model):
        assert all(m in current_metrics for m in self.monitor_metrics), "Missing monitored metrics in current_metrics"

        improvements = {m: self._is_improved(m, current_metrics[m]) for m in self.monitor_metrics}

        if self.stop_mode == "any":
            improved = any(improvements.values())
        elif self.stop_mode == "all":
            improved = all(improvements.values())
        else:
            raise ValueError("stop_mode must be either 'any' or 'all'")

        if improved:
            for m in self.monitor_metrics:
                if improvements[m]:
                    self.best_metrics[m] = current_metrics[m]
            self.counter = 0
            self.best_model_wts = model.state_dict()
            if self.path:
                torch.save(model.state_dict(), self.path)
                if self.verbose:
                    print(f"Improvement in {improvements}. Saved model to {self.path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

    def load_best_model(self, model):
        if self.best_model_wts:
            model.load_state_dict(self.best_model_wts)
        elif self.path:
            model.load_state_dict(torch.load(self.path))
