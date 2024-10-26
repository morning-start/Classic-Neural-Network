import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


class UtilNet:
    def __init__(
        self, model: nn.Module, metrics: dict = {}, device=torch.device("cpu")
    ):
        self.model = model
        self.device = device
        self.metrics = metrics
        self.losses = []
        self.history: dict[str, list] = {}
        self.metrics_results: dict[str, list] = {}
        self.model.to(self.device)

        for metric_name in self.metrics:
            self.history[metric_name] = []

    def _reduce_metric(
        self,
        metric,
        output: list[np.ndarray],
        target: list[np.ndarray],
    ):
        res = map(metric, output, target)
        res = np.mean(list(res))
        return res

    def _calculate_metrics(self, y_pred, y_true):
        metrics_results: dict[str, list] = {}

        for metric_name in self.metrics:
            metrics_results[metric_name] = []

        for metric_name, metric in self.metrics.items():
            res = self._reduce_metric(metric, y_pred, y_true)
            metrics_results[metric_name].append(res)
        return metrics_results

    def train(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
    ):
        """Train the network on the training set."""
        for epoch in range(num_epochs):
            self.model.train()
            y_pred = []
            y_true = []
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output: torch.Tensor = self.model(data)
                loss: torch.Tensor = criterion(output, target)
                loss.backward()
                optimizer.step()

                y_pred.append(output.detach().cpu().numpy())
                y_true.append(target.cpu().numpy())

            self.losses.append(loss.item())
            res = self._calculate_metrics(y_pred, y_true)

            for metric_name in res:
                self.history[metric_name].extend(res[metric_name])

    def test(self, test_loader: DataLoader) -> dict:
        """Test the network on the test set."""
        self.model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                y_pred.append(output.detach().cpu().numpy())
                y_true.append(target.cpu().numpy())

        res = self._calculate_metrics(y_pred, y_true)
        for metric_name in res:
            self.metrics_results[metric_name] = res[metric_name][0]
