from .abstract_policy import AbstractPolicy
import torch
import numpy as np
from src.env.environment import Environment


class NeuralNetPolicy(AbstractPolicy):
    """
    Abstract class of any policy

    This class is meant for programming purposes and can't be instantiated
    """

    def __init__(
            self,
            environment: Environment,
            hidden_size: int = 500,
            learning_rate: float = 0.003,
    ):
        super().__init__(environment)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.environment.n_observations, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.environment.n_actions),
            torch.nn.Softmax(dim=0)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict_proba(self, state: np.array):
        return self.model(torch.from_numpy(state).float())

    def learn_one(self, transitions, gamma):
        reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))
        batch_Gvals = []
        for i in range(len(transitions)):
            new_Gval = 0
            power = 0
            for j in range(i, len(transitions)):
                new_Gval = new_Gval + ((gamma ** power) * reward_batch[j]).numpy()
                power += 1
            batch_Gvals.append(new_Gval)
        expected_returns_batch = torch.FloatTensor(batch_Gvals)
        expected_returns_batch /= expected_returns_batch.max()
        state_batch = torch.Tensor([s for (s, a, r) in transitions])
        action_batch = torch.Tensor([a for (s, a, r) in transitions])
        pred_batch = self.model(state_batch)
        prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()

        loss = -torch.sum(torch.log(prob_batch) * expected_returns_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
