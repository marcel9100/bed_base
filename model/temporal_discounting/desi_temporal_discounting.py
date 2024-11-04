"""
Working with Stan in python

1. Install cmdstanpy

`conda install -c conda-forge cmdstanpy`

You might need to install cython as well (`pip install Cython`) - try wihtout first

2. Install cmdstan from python:

import cmdstanpy
cmdstanpy.install_cmdstan()

"""

#### Delay Discounting Task ####
"""
This module implements a delay discounting task model.

Key Components:
1. Outcome: Binary choice between Larger-Later (LL) or Smaller-Sooner (SS) rewards.
2. Subjective Value: V = R * d(t), where:
   - V: subjective value
   - R: reward amount
   - d(t): discount function
   - t: time delay

3. Discount Function: d(t) = 1 / (1 + k*t), where k is the discount rate

4. Design Variables:
   - t: Time delay for the LL reward
   - R_SS: Reward amount for the SS option (immediate reward)
   - R_LL: Fixed at 1.0 (normalized larger-later reward)
   - t_SS: Fixed at 0.0 (immediate smaller-sooner delay)

5. Model Parameters (θ):
   - k: Discount rate, where log(k) ~ N(-4.25, 1.5)
   - α: Choice sensitivity, where α ~ HalfNormal(0, 2) + 1e-3

6. Choice Probability:
   P(y=LL | k, α, R, t) = ε + (1-2ε) * Φ((V_LL - V_SS) / α)
   Where:
   - Φ: Cumulative distribution function of the standard normal distribution
   - ε: Fixed at 0.01 (to account for lapses in attention)

7. Outcome Distribution:
   y ~ Bernoulli(p), where p is the probability of choosing LL
"""

import math
import copy
import os
import tempfile
import json
from typing import Callable
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor
import torch.nn.functional as F

#from cmdstanpy import CmdStanModel

import matplotlib.pyplot as plt

import mlflow
import mlflow.pytorch as mlflow_pytorch


# swith on tracking/debugging of autograd
# torch.autograd.set_detect_anomaly(True)


class TemporalDiscounting(nn.Module):
    def __init__(
        self,
        design_net: nn.Module,
        prior_log_k: dist.Distribution,
        prior_alpha: dist.Distribution,
        T: int,
        shift: float = 0.0,
        epsilon: float = 0.01,
        long_reward: float = 100.0,
        discount_type: str = "hyperbolic",
    ):
        """
        Args:
            design_net: nn.Module
                The design network to use.
            prior_log_k: dist.Distribution
                The prior distribution for the log of the discount rate.
            prior_alpha: dist.Distribution
                The prior distribution for the choice sensitivity.
            T: int
                The number of time steps.
            shift: float
                Shift parameter for design transformation.
            epsilon: float
                Epsilon value for outcome likelihood calculation.
            long_reward: float
                The reward for the larger-later option.
            discount_type: str
                The type of discounting to use ("hyperbolic" or "exponential").
        """
        super().__init__()
        self.design_net = design_net
        self.T = T
        self.shift = shift
        self.prior_log_k = prior_log_k
        self.prior_alpha = prior_alpha
        self.discount_type = discount_type

        self.register_buffer("short_delay", torch.tensor(0.0))
        self.register_buffer("long_reward", torch.tensor(long_reward))
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def transform_designs(self, designs: Tensor) -> tuple[Tensor, Tensor]:
        """Transform the designs to the constrained space."""
        reward_short_delay, delay = designs.unbind(-1)  # [B], [B]
        delay = (delay - self.shift).exp()
        reward_short_delay = self.long_reward * torch.sigmoid(reward_short_delay)
        return reward_short_delay, delay

    def get_present_value(
        self, log_k: Tensor, designs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Get the present value of the LL and SS options.
        """
        k = log_k.exp()
        reward_short_delay, delay = self.transform_designs(designs)  # [B], [B]
        if self.discount_type == "hyperbolic":  # reward / (1 + k * delay)
            present_value_LL = self.long_reward / (1.0 + k * delay)  # [B]
            present_value_SS = reward_short_delay / (1.0 + k * self.short_delay)  # [B]
        elif self.discount_type == "exponential":  # reward * exp(-k * delay)
            present_value_LL = self.long_reward * (-k * delay).exp()  # [B]
            present_value_SS = reward_short_delay * (-k * self.short_delay).exp()  # [B]
        else:
            raise ValueError(f"Unknown discount type: {self.discount_type}")
        return present_value_LL, present_value_SS

    def outcome_likelihood(
        self, log_k: Tensor, alpha: Tensor, designs: Tensor
    ) -> dist.Distribution:
        """
        Calculate the likelihood of the outcome given the parameters and designs.

        Args:
            log_k: Tensor of shape [B], log of the discount rate.
            alpha: Tensor of shape [B], choice sensitivity.
            designs: Tensor of shape [B, 2], untransformed designs.
                * B is the batch size (can be B or [N, B])

        Returns:
            A Bernoulli distribution representing the likelihood of choosing LL.
        """
        present_value_LL, present_value_SS = self.get_present_value(log_k, designs)
        diff = present_value_LL - present_value_SS  # [B]

        # probability of selecting LL
        p = self.epsilon + (1.0 - 2.0 * self.epsilon) * (
            0.5 + 0.5 * torch.erf(diff / (alpha.abs() + 1e-3))
        )  # [B]
        # with torch.no_grad():
        #     erf_diff = torch.erf(diff) - torch.erf(diff / (alpha + 1e-3))
        #     print("??", erf_diff.max(), erf_diff.min(), erf_diff.mean())

        p = p.unsqueeze(-1)  # [B] -> [B, 1]
        return dist.Bernoulli(probs=p)

    def forward(
        self,
        batch_size: int = 1,
        # pass past designs-outcomes to condition on, if any
        past_designs: Tensor | None = None,
        past_outcomes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        log_k = self.prior_log_k.sample(torch.Size([batch_size]))  # [B]
        alpha = self.prior_alpha.sample(torch.Size([batch_size]))  # [B]

        designs = torch.empty(
            (self.T, batch_size, self.design_net.design_shape),
            dtype=torch.float32,
        )  # [T, B, 2]
        outcomes = torch.empty((self.T, batch_size, 1), dtype=torch.float32)
        num_past_datapoints = 0

        if past_designs is not None and past_designs.shape[0] > 0:
            # concat the past_designs to designs [T, B, 2]
            # assert past_outcomes is not None and the T-s match
            num_past_datapoints = past_designs.shape[0]
            assert (
                past_outcomes is not None
                and past_outcomes.shape[:-1] == past_designs.shape[:-1]
            )
            designs = torch.cat([past_designs, designs], dim=0)  # [T+past, B, 2]
            outcomes = torch.cat([past_outcomes, outcomes], dim=0)  # [T+past, B, 1]

        for t in range(num_past_datapoints, self.T + num_past_datapoints):
            #! clone to avoid inplace modification!
            designs[t] = self.design_net(
                designs[:t].clone(), outcomes[:t].clone()
            )  # [B, 2]
            outcomes[t] = self.outcome_likelihood(
                log_k, alpha, designs[t]
            ).sample()  # [B, 1]
        # return only the "new" designs and outcomes
        return (
            (log_k, alpha),
            designs[num_past_datapoints:],
            outcomes[num_past_datapoints:],
        )

    @torch.no_grad()
    def run_policy(
        self,
        log_k: Tensor,
        alpha: Tensor,
        past_designs: Tensor | None = None,
        past_outcomes: Tensor | None = None,
        return_probs: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Run a trajectory of T designs for the given parameters."""
        batch_size = log_k.shape[0]
        designs = torch.empty(
            (self.T, batch_size, self.design_net.design_shape),
            dtype=torch.float32,
        )  # [T, B, 2]
        outcomes = torch.empty(
            (self.T, batch_size, 1), dtype=torch.float32
        )  # [T, B, 1]
        num_past_datapoints = 0

        if past_designs is not None and past_designs.shape[0] > 0:
            assert past_outcomes is not None
            num_past_datapoints = past_designs.shape[0]
            designs = torch.cat([past_designs, designs], dim=0)
            outcomes = torch.cat([past_outcomes, outcomes], dim=0)

        for t in range(num_past_datapoints, self.T + num_past_datapoints):
            designs[t] = self.design_net(
                designs[:t].clone(), outcomes[:t].clone()
            )  # [B, 2]
            if return_probs:
                outcomes[t] = self.outcome_likelihood(
                    log_k, alpha, designs[t]
                ).probs  # [B, 1]
            else:
                outcomes[t] = self.outcome_likelihood(
                    log_k, alpha, designs[t]
                ).sample()  # [B, 1]

        # [T, B, 2], [T, B, 1]
        return designs[num_past_datapoints:], outcomes[num_past_datapoints:]

    def plot_diff(
        self,
        log_k: Tensor,
        alpha: Tensor,
        past_designs: Tensor | None = None,
        past_outcomes: Tensor | None = None,
        save_path: str | None = None,
    ):
        """Plot the difference in present value vs design number for batches of parameters."""
        batch_size = log_k.shape[0]
        plt.figure(figsize=(7, 5))

        for realization in range(5):
            # we sample 5 different trajectories for each parameter;
            # this is to show the variability in the outcome (Ber(p))
            designs, outcomes = self.run_policy(
                log_k, alpha, past_designs, past_outcomes
            )  # [T, B, 2], [T, B, 1]

            present_value_LL, present_value_SS = self.get_present_value(log_k, designs)
            diff = present_value_LL - present_value_SS

            for i in range(batch_size):
                for t in range(self.T):
                    color = "green" if outcomes[t, i].item() == 1 else "black"
                    plt.scatter(
                        t + realization * 0.1,
                        diff[t, i],
                        color=color,
                        label=(
                            f"Param set {i+1}" if t == 0 and realization == 0 else ""
                        ),
                    )

        plt.axhline(0, linestyle="--", color="gray")
        plt.xlabel("Design number")
        plt.ylabel("Difference in present value")
        plt.title("Present Value Difference across 5 policy rolls")
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close("all")

    def plot_designs_history(
        self, designs_history: Tensor, save_path: str | None = None
    ):
        """Plot the history of design values over gradient steps."""
        num_steps, T, batch_size, _ = designs_history.shape  # [num_steps, T, B, 2]

        fig, axs = plt.subplots(
            2, (T + 1) // 2, figsize=(15, 10), sharex=True, sharey=True
        )
        fig.suptitle("Design Values vs Gradient Steps")
        axs = axs.flatten()

        x = range(num_steps)
        colors = plt.cm.viridis(torch.linspace(0, 1, batch_size))

        for t in range(self.T):
            for b in range(batch_size):
                reward_short_delay = [
                    designs[t, b, 0].item() for designs in designs_history
                ]
                delay = [designs[t, b, 1].item() for designs in designs_history]

                axs[t].plot(
                    x,
                    reward_short_delay,
                    color=colors[b],
                    label=f"SS b={b+1}",
                )
                axs[t].plot(
                    x, delay, color=colors[b], linestyle="--", label=f"Delay b={b+1}"
                )
            axs[t].set_title(f"Design {t+1}")
            axs[t].set_ylabel(f"Design {t+1} Value")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        axs[-1].set_xlabel("Gradient Step")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close("all")


class NestedMonteCarlo(nn.Module):

    def __init__(
        self,
        prior_log_k: dist.Distribution,
        prior_alpha: dist.Distribution,
        outcome_likelihood: Callable,  # should return a distribution when called with params and designs
        proposal_log_k: dist.Distribution | None = None,
        proposal_alpha: dist.Distribution | None = None,
        num_inner_samples: int = 16,
        lower_bound: bool = True,
        auto_proposal: bool = False,  # not implemented yet
    ):
        """
        Args:
            prior_log_k: dist.Distribution
                Prior distribution for the log of the discount rate.
            prior_alpha: dist.Distribution
                Prior distribution for the choice acuity.
            outcome_likelihood: Callable
                Outcome likelihood function - should return a distribution when called with params and designs
            proposal_log_k: dist.Distribution | None
                Importance sampling proposal distribution for the log of the discount;
                Optional, if None, we use the prior
            proposal_alpha: dist.Distribution | None
                Importance sampling proposal distribution for the choice acuity;
                Optional, if None, we use the prior
            num_inner_samples: int
                The number of samples to use to approximate the marginal.
            lower_bound: bool
        """
        super().__init__()
        self.prior_log_k = prior_log_k
        self.prior_alpha = prior_alpha
        self.proposal_log_k = proposal_log_k
        self.proposal_alpha = proposal_alpha
        # self.auto_proposal = auto_proposal

        self.outcome_likelihood = outcome_likelihood
        self.num_inner_samples = num_inner_samples
        self.lower_bound = lower_bound

    def get_log_likelihood_and_marginal(
        self, log_k: Tensor, alpha: Tensor, designs: Tensor, outcomes: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch_size = log_k.shape[0]  # log_k should be shape [B]
        T = designs.shape[0]  # designs should be shape [T, B, 2]
        log_likelihood = torch.stack(  # [B, 1]
            [
                self.outcome_likelihood(log_k, alpha, xi).log_prob(y)
                for xi, y in zip(designs, outcomes)
            ],
            dim=0,
        ).sum(dim=0)
        # inner_log_k and inner_alpha are [N, B]
        inner_log_k = self.prior_log_k.sample((self.num_inner_samples, batch_size))
        inner_alpha = self.prior_alpha.sample((self.num_inner_samples, batch_size))

        inner_log_likelihood = torch.stack(  # [N, B, 1]
            [
                self.outcome_likelihood(inner_log_k, inner_alpha, xi).log_prob(y)
                for xi, y in zip(designs, outcomes)
            ],
            dim=0,
        ).sum(dim=0)

        if self.lower_bound:
            inner_log_likelihood = torch.cat(
                [log_likelihood.unsqueeze(0), inner_log_likelihood], dim=0
            )  # [N+1, B, 1]
        # log-sum rather than log-mean-exp. constant added back in .estimate only
        log_marginal = torch.logsumexp(inner_log_likelihood, dim=0)  # [B, 1]
        return log_likelihood, log_marginal

    @torch.no_grad()
    def estimate(
        self, log_k: Tensor, alpha: Tensor, designs: Tensor, outcomes: Tensor
    ) -> float:
        """
        Estimate the mutual information between params and outcomes (EIG)
        I(params; outcomes) = \E_{p(params)p(outcomes|params, designs)} [
            log p(outcomes|params, designs) - log p(outcomes|designs)
        ]

        Note that this is an expectation over the parameters and designs.
        The designs are being optimised over so we need to be careful with the expectation -->
            - Reparameterisation trick where .has_rsample() is True
            - REINFORCE gradient estimator otherwise

        params: Tensor of shape [B, 2]
        designs: Tensor of shape [T, B, 2]
        outcomes: Tensor of shape [T, B, 1]

        returns: float
        """
        log_likelihood, log_marginal = self.get_log_likelihood_and_marginal(
            log_k, alpha, designs, outcomes
        )
        mi_estimate = (log_likelihood - log_marginal).mean(0) + math.log(
            self.num_inner_samples + self.lower_bound
        )
        return mi_estimate.item()

    def differentiable_loss(
        self, log_k: Tensor, alpha: Tensor, designs: Tensor, outcomes: Tensor
    ) -> Tensor:
        # implement the differentiable loss using REINFORCE (aka score) gradient estimator
        log_likelihood, log_marginal = self.get_log_likelihood_and_marginal(
            log_k, alpha, designs, outcomes
        )
        ## REINFORCE gradient estimator
        mi_estimate_no_grad = (log_likelihood - log_marginal).detach()
        diff_loss = -(mi_estimate_no_grad * log_likelihood - log_marginal).mean(0)
        return diff_loss


class StaticDesignNetwork(nn.Module):
    def __init__(self, design_shape: torch.Size, T: int, designs: Tensor | None = None):
        super().__init__()
        self.design_shape = design_shape
        self.T = T
        if designs is None:
            designs = nn.Parameter(torch.rand(T, design_shape) * 2 - 1.0)
        else:
            designs = nn.Parameter(designs)
        self.register_parameter("designs", designs)

    def forward(self, designs: Tensor, outcomes: Tensor) -> Tensor:
        t = designs.shape[0]
        batch_size = designs.shape[1]
        return self.designs[t].expand(batch_size, self.design_shape)


class DeepAdaptiveDesign(nn.Module):
    def __init__(
        self,
        design_shape: int,
        T: int,
        y_dim: int = 1,
        hidden_dim: int = 128,
        embedding_dim: int = 16,
        time_embedding: bool = True,
    ):
        super().__init__()
        self.design_shape = design_shape
        self.T = T
        self.encode_designs = nn.Sequential(
            nn.Linear(design_shape, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )
        self.head0 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )
        self.head1 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )
        self.decode_designs = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, design_shape),
        )
        self.time_embedding = time_embedding
        if time_embedding:
            self.time_projection = nn.Linear(embedding_dim * 2, embedding_dim)
            # params T of dim embedding_dim
            self.register_parameter(
                "time_embeddings", nn.Parameter(torch.rand(T, embedding_dim))
            )

    def forward(self, designs: Tensor, outcomes: Tensor) -> Tensor:
        t = designs.shape[0]
        batch_size = designs.shape[1]

        # Encode designs and outcomes
        encoded = self.encode_designs(designs)
        x_0 = self.head0(encoded)
        x_1 = self.head1(encoded)
        x = outcomes * x_1 + (1.0 - outcomes) * x_0

        # Sum over time steps
        x = x.sum(dim=0)  # [B, embedding_dim]
        if self.time_embedding:
            # self.time_embeddings[t] is [embedding_dim]
            time_embedding = self.time_embeddings[t].expand(*x.shape)
            # concat time embedding with the summed features
            x = torch.cat([x, time_embedding], dim=-1)
            x = self.time_projection(x)

        return self.decode_designs(x)


def train_amortized_model(
    T: int = 20,
    design_arch: str = "static",
    lr: float = 5e-3,
    batch_size: int = 64,
    num_inner_samples: int = 128,
    lower_bound: bool = True,
    num_grad_steps: int = 1000,
    seed: int = 11,
):
    torch.manual_seed(seed)
    if design_arch == "static":
        design_net = StaticDesignNetwork(design_shape=2, T=T)
    elif design_arch == "dad":
        design_net = DeepAdaptiveDesign(
            design_shape=2, T=T, y_dim=1, hidden_dim=128, embedding_dim=32
        )
    else:
        raise ValueError(f"Unknown design architecture: {design_arch}")

    # log_k is Normal; alpha is abs(Normal)
    log_k_dist = dist.Normal(torch.tensor(-4.25), torch.tensor(1.5))
    alpha_dist = dist.Normal(torch.tensor(0.0), torch.tensor(2.0))

    shift = -4.25  # + 0.5 * 1.5**2  # (mean + 0.5 * var) to shift the delay
    model = TemporalDiscounting(
        design_net=design_net,
        T=T,
        prior_log_k=log_k_dist,
        prior_alpha=alpha_dist,
        shift=shift,
        discount_type="hyperbolic",
    )

    (log_k, alpha), d1, o1 = model(7)
    assert log_k.shape == (7,)
    assert alpha.shape == (7,)
    assert d1.shape == (T, 7, 2)
    assert o1.shape == (T, 7, 1)

    objective = NestedMonteCarlo(
        prior_log_k=log_k_dist,
        prior_alpha=alpha_dist,
        outcome_likelihood=model.outcome_likelihood,
        num_inner_samples=num_inner_samples,
        lower_bound=True,
    )
    mi_eval_lower = NestedMonteCarlo(
        prior_log_k=log_k_dist,
        prior_alpha=alpha_dist,
        outcome_likelihood=model.outcome_likelihood,
        num_inner_samples=10000,
        lower_bound=True,
    )
    mi_eval_upper = NestedMonteCarlo(
        prior_log_k=log_k_dist,
        prior_alpha=alpha_dist,
        outcome_likelihood=model.outcome_likelihood,
        num_inner_samples=10000,
        lower_bound=False,
    )

    mlflow.set_experiment("delay_discounting")
    with mlflow.start_run():
        mlflow.log_param("T", T)
        mlflow.log_param("design_arch", design_arch)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_inner_samples", num_inner_samples)
        mlflow.log_param("lower_bound", lower_bound)
        mlflow.log_param("num_grad_steps", num_grad_steps)
        mlflow.log_param("seed", seed)

        res_mi_lower, res_mi_upper = zip(
            *[
                (
                    mi_eval_lower.estimate(log_k, alpha, designs, outcomes),
                    mi_eval_upper.estimate(log_k, alpha, designs, outcomes),
                )
                for (log_k, alpha), designs, outcomes in (model(32) for _ in range(256))
            ]
        )
        res_mi_lower = torch.tensor(res_mi_lower)
        res_mi_upper = torch.tensor(res_mi_upper)

        print(f"MI (lower bound): {res_mi_lower.mean()}")
        print(f"MI (upper bound): {res_mi_upper.mean()}\n")

        mlflow.log_metric("eval_MI_lower", res_mi_lower.mean().item(), step=0)
        mlflow.log_metric("eval_MI_upper", res_mi_upper.mean().item(), step=0)

        # plot the diff = VLL - VSS on the y axis vs design number on the x
        test_log_k, test_alpha = (
            log_k_dist.sample(torch.Size([128])),
            alpha_dist.sample(torch.Size([128])),
        )  # [128], [128]
        with tempfile.NamedTemporaryFile(
            prefix="pvdiff_before_", suffix=".png"
        ) as tmpf:
            model.plot_diff(test_log_k[:10], test_alpha[:10], save_path=tmpf.name)
            mlflow.log_artifact(tmpf.name, artifact_path="plots")

        pbar = trange(num_grad_steps)
        logging_freq = 100 if num_grad_steps > 500 else 10
        estimates_history = []

        optimiser = torch.optim.AdamW(design_net.parameters(), lr=lr, weight_decay=1e-4)
        # cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=num_grad_steps, eta_min=1e-6
        )
        lrs = []
        designs_history = []
        for i in pbar:
            # store the current learning rate
            lrs.append(optimiser.param_groups[0]["lr"])
            model.train()
            optimiser.zero_grad()
            (log_k, alpha), designs, outcomes = model(batch_size)

            loss = objective.differentiable_loss(log_k, alpha, designs, outcomes)
            if i % logging_freq == 0:
                # run the policy on the test params
                test_designs, test_outcomes = model.run_policy(test_log_k, test_alpha)
                designs_history.append(test_designs[:, :3, :].cpu())  # keep only 3
                mlflow.log_metric("train_loss", loss.item(), step=i)
                mi_estimate = mi_eval_lower.estimate(
                    test_log_k, test_alpha, test_designs, test_outcomes
                )
                mlflow.log_metric("train_MI_lower", mi_estimate, step=i)
                pbar.set_description(f"Loss: {loss.item():.3f} MI: {mi_estimate:.3f}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()

        model.eval()

        with tempfile.NamedTemporaryFile(prefix="pvdiff_after_", suffix=".png") as tmpf:
            model.plot_diff(test_log_k[:10], test_alpha[:10], save_path=tmpf.name)
            mlflow.log_artifact(tmpf.name, artifact_path="plots")

        designs_history_tensor = torch.stack(designs_history)
        with tempfile.NamedTemporaryFile(prefix="designs_hist_", suffix=".png") as tmpf:
            model.plot_designs_history(designs_history_tensor, save_path=tmpf.name)
            mlflow.log_artifact(tmpf.name, artifact_path="plots")

        with tempfile.NamedTemporaryFile(prefix="designs_hist_", suffix=".pt") as tmpf:
            torch.save(designs_history_tensor, tmpf.name)
            mlflow.log_artifact(tmpf.name, artifact_path="pt")

        # learning rate vs iteration
        plt.figure(figsize=(10, 6))
        plt.plot(lrs)
        plt.xlabel("Iteration (x100)")
        plt.ylabel("Learning rate")
        plt.title("Learning rate vs Iteration")
        with tempfile.NamedTemporaryFile(prefix="lr_vs_iter_", suffix=".png") as tmpf:
            plt.savefig(tmpf.name)
            mlflow.log_artifact(tmpf.name, artifact_path="plots")
        plt.close("all")

        # evaluate MI after optimising
        res_mi_lower, res_mi_upper = zip(
            *[
                (
                    mi_eval_lower.estimate(log_k, alpha, designs, outcomes),
                    mi_eval_upper.estimate(log_k, alpha, designs, outcomes),
                )
                for (log_k, alpha), designs, outcomes in (model(32) for _ in range(256))
            ]
        )
        res_mi_lower = torch.tensor(res_mi_lower)
        res_mi_upper = torch.tensor(res_mi_upper)

        print(f"After training MI (lower bound): {res_mi_lower.mean()}")
        print(f"After training MI (upper bound): {res_mi_upper.mean()}\n")

        mlflow.log_metric("eval_MI_lower", res_mi_lower.mean().item(), step=1)
        mlflow.log_metric("eval_MI_upper", res_mi_upper.mean().item(), step=1)
        mi_results = {
            "eval_MI_lower": res_mi_lower.tolist(),
            "eval_MI_upper": res_mi_upper.tolist(),
        }
        with tempfile.NamedTemporaryFile(
            prefix="eval_MI_", suffix=".json", mode="w"
        ) as tmpf:
            json.dump(mi_results, tmpf)
            tmpf.flush()
            mlflow.log_artifact(tmpf.name, artifact_path="json")

        # Log the model
        mlflow_pytorch.log_model(model, "model")

    return model


class EmpiricalPrior:
    def __init__(self, samples: Tensor):
        self.samples = samples
        self.N = len(self.samples)

    def sample(self, shape: torch.Size) -> Tensor:
        # sample randomly from the given samples
        idx = torch.randint(0, self.N, shape)
        return self.samples[idx]


################ MAIN #########################
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        breakpoint()
    lr = 1e-4
    T = 20
    design_arch = "dad"
    num_grad_steps_list = [10000]  # 10_000, 50_000
    models = {}
    for num_grad_steps in num_grad_steps_list:
        print(f"Training with {num_grad_steps} gradient steps")
        models[num_grad_steps] = model = train_amortized_model(
            T=T,
            design_arch=design_arch,
            lr=lr,
            num_grad_steps=num_grad_steps,
            seed=11,
            batch_size=256,
            num_inner_samples=512,
        ).to(device)
    # # load the model from mlflow:
    # # model = mlflow.pytorch.load_model("runs:/f17a6d566a014044b29bcbc1e998aae9/model")
    # # run inference
    # # pretend we are doing a real experiment, pick one of the trained models
    # model = models[5000]
    # torch.manual_seed(20240923)
    # real_log_k = model.prior_log_k.sample(torch.Size([1]))
    # real_alpha = model.prior_alpha.sample(torch.Size([1]))

    # real_designs, real_outcomes = model.run_policy(real_log_k, real_alpha)
    # # real designs is [T, B, 2], B = 1
    # # real outcomes is [T, B, 1], B = 1
    # stan_model = CmdStanModel(stan_file="htd.stan")
    # stan_data = {
    #     "T": model.T,  # +1 because we are indexing from 0
    #     "reward_short_delay": real_designs[:, 0, 0].numpy().tolist(),
    #     "delay": real_designs[:, 0, 1].numpy().tolist(),
    #     "long_reward": model.long_reward.item(),
    #     "epsilon": model.epsilon.item(),
    #     "shift": model.shift,
    #     "y": real_outcomes[:, 0, 0].numpy().astype(int).tolist(),
    # }
    # fit = stan_model.sample(
    #     data=stan_data,
    #     chains=4,
    #     iter_sampling=2500,
    #     iter_warmup=10000,
    #     show_progress=False,
    #     max_treedepth=20,
    #     adapt_delta=0.99,
    # )
    # posterior_samples = fit.draws_pd()
    # log_k_post_samples = torch.tensor(posterior_samples["log_k"], dtype=torch.float32)
    # alpha_post_samples = torch.tensor(
    #     posterior_samples["alpha_untransformed"], dtype=torch.float32
    # )
    # # plot the posterior as a scatter plot, with the true sample as red cross
    # plt.figure(figsize=(10, 6))
    # plt.scatter(
    #     model.prior_log_k.sample(torch.Size([10000])),
    #     model.prior_alpha.sample(torch.Size([10000])),
    #     color="black",
    #     alpha=0.1,
    #     label="Prior samples",
    # )
    # plt.scatter(
    #     log_k_post_samples,
    #     alpha_post_samples,
    #     alpha=0.05,
    #     s=3,
    #     label="HMC posterior samples",
    # )
    # plt.scatter(
    #     real_log_k, real_alpha, color="red", marker="x", s=100, label="True parameter"
    # )
    # plt.xlabel("log_k")
    # plt.ylabel("alpha")
    # plt.legend()
    # plt.title("HTD model: Posterior samples")
    # plt.show()
