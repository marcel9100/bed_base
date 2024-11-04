import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor
import math
from typing import Callable

import mlflow

from tqdm import trange

from design_arch.dad import Dad, DeepAdaptiveDesign

import numpy as np

class TemporalDiscounting(nn.Module):
    def __init__(
            self,
            device,
            design_net: nn.Module, # design network
            log_k_loc: float, # prior mean of log(k) dist
            log_k_scale: float, # prior sd of log(k) dist
            prior_alpha: dist.Distribution, # prior distribution of alpha
            T=2,
            short_delay=0.0,
            long_reward=100.0,
            epsilon: float=0.01
            ):
        super().__init__()
        self.design_net = design_net
        self.log_k_loc = log_k_loc
        self.log_k_scale = log_k_scale
        self.prior_alpha_dist = prior_alpha
        self.T = T
        self.short_delay = short_delay
        self.long_reward = long_reward
        self.epsilon = epsilon
        self.sigmoid = nn.Sigmoid()
        self.device = device

        #set up the log(k) prior 
        self.prior_log_k_dist = dist.Normal(loc=torch.tensor([log_k_loc],device=device), scale=torch.tensor([log_k_scale],device=device))

    def transform_xi(self, xi, shift=0.0):

        short_reward, delay  = xi[..., 0:1], xi[..., 1:2] #[B,1], [B,1]
        delay = (delay - shift).exp() # [B,1]

        short_reward = self.long_reward * self.sigmoid(short_reward) # [B,1]
        return short_reward, delay
        
    def outcome_likelihood(
        self,
        log_k, # [B, 1]
        alpha, # [B, 1]
        design, # [B, 2]
        )-> dist.Distribution:
        """
        Compute the likelihood of the outcome given the design

        Args:
            log_k: 
                [B,1] - log of discount rate
            alpha: 
                [B,1] - sensitivity
            design: 
                [B, 2]
        Returns:
            dist.Distribution - bernouli
        """
        # Use this as an offset to help with initialization
        log_k_mean = self.log_k_loc + 0.5*self.log_k_scale*self.log_k_scale #float

        

        short_reward, delay = self.transform_xi(design,shift=log_k_mean) # [B,1], [B,1]

        v_a = short_reward/(1+torch.exp(log_k)*self.short_delay) #[B,1]
        v_b = self.long_reward/(1+torch.exp(log_k)*delay) #[B,1]
        erf_arg = (v_a - v_b)/(alpha.abs() + 1e-3) #[B,1]

        psi = self.epsilon + (1.0 - 2.0 * self.epsilon) * (0.5 + 0.5 * torch.erf(erf_arg)) # [B, 1]

        return dist.Bernoulli(probs=psi)
    
    def forward(self,
                batch_size:int,
                past_designs: Tensor | None = None,
                past_outcomes: Tensor | None = None,
                ) -> tuple[Tensor, Tensor, Tensor]:        
        """
        Forward pass of the model

        Args:
            batch_size: int
                The number of batches
            past_designs: torch.Tensor
                The design matrix of shape [batch_size,t, 2]
            past_outcomes: torch.Tensor
                The outcome matrix of shape [batch_size,t, 1]

        Returns:
            log_k: torch.Tensor
                The sampled log_k of shape [batch_size, 1]
            alpha: torch.Tensor
                The sampled alpha of shape [batch_size, 1]
            design: torch.Tensor
                The design matrix of shape [batch_size,t+num_past_datapoints, 2]
            outcome: torch.Tensor
                The outcome matrix of shape [batch_size,t+num_past_datapoints, 1]
        """
        # Sample the prior
        log_k = self.prior_log_k_dist.sample([batch_size]).to(self.device) # [batch_size,1]
        alpha = self.prior_alpha_dist.sample([batch_size]).to(self.device) # [batch_size,1]
        

        if past_designs is None:
            num_past_datapoints = 0
        else:
            num_past_datapoints = past_designs.shape[1]
            assert past_outcomes.shape[:-1] == past_designs.shape[:-1] #same shape up till the last dimension

        #set up empty design and outcome tensors
        designs = torch.empty(batch_size, 0, 2, dtype=torch.float32, device=self.device) if past_designs is None else past_designs
        outcomes = torch.empty(batch_size, 0, 1,dtype=torch.float32, device=self.device) if past_outcomes is None else past_outcomes

        for t in range(num_past_datapoints, self.T+num_past_datapoints):
            xi = self.design_net(designs, outcomes).to(self.device) # [batch_size, design_dim]
            yi = self.outcome_likelihood(log_k, alpha, xi).sample().to(self.device) # [batch_size, observation_dim]

            designs = torch.cat([designs, xi.unsqueeze(1)], dim=1) # [batch_size, t+1, design_dim]
            outcomes = torch.cat([outcomes, yi.unsqueeze(1)], dim=1) # [batch_size, t+1, observation_dim]

        return log_k, alpha, designs, outcomes #[B,1], [B,1], [B,T,2], [B,T,1]
    
    @torch.no_grad()
    def run_policy(
            self,
            log_k, # [B, 1]
            alpha, # [B, 1]
            past_designs: Tensor | None = None,
            past_outcomes: Tensor | None = None,
            ) -> tuple[Tensor, Tensor]:
            """
            Run a trajectory of T designs for the given latent parameters
            """

            batch_size = log_k.shape[0]

            if past_designs is None:
                num_past_datapoints = 0
            else:
                num_past_datapoints = past_designs.shape[1]
                assert past_outcomes.shape[:-1] == past_designs.shape[:-1] #same shape up till the last dimension

            #set up empty design and outcome tensors
            designs = torch.empty(batch_size, 0, 2,dtype=torch.float32, device=self.device) if past_designs is None else past_designs
            outcomes = torch.empty(batch_size, 0, 1, dtype=torch.float32,device=self.device) if past_outcomes is None else past_outcomes

            for t in range(num_past_datapoints, self.T+num_past_datapoints):
                xi = self.design_net(designs, outcomes, num_past_datapoints).to(self.device) # [batch_size, design_dim]
                yi = self.outcome_likeihood(log_k, alpha, xi).sample() # [batch_size, observation_dim]

                designs = torch.cat([designs, xi.unsqueeze(1)], dim=1) # [batch_size, t+1, design_dim]
                outcomes = torch.cat([outcomes, yi.unsqueeze(1)], dim=1) # [batch_size, t+1, observation_dim]

            return designs, outcomes
    
#####################################################################################################################################################
class PriorContrastiveEstimation(nn.Module):
    def __init__(
        self,
        model,
        outcome_likelihood: Callable,  # should return a distribution when called with params and designs
        num_inner_samples: int = 16,
        lower_bound: bool = True,
    ):
        super().__init__()
        self.model = model
        self.outcome_likelihood = outcome_likelihood
        self.num_inner_samples = num_inner_samples
        self.lower_bound = lower_bound
    
    def log_likelihood(self, log_k, alpha,designs, outcomes):
        """
        Args:
            log_k: Tensor
                The log_k parameter. [B, 1] or [L+1,B,1]
            alpha: Tensor
                The alpha parameter. [B, 1] or [L+1,B,1]
            designs: Tensor
                The designs. [B, T, 2]
            outcomes: Tensor
                The outcomes. [B, T, 1]
        
        Returns:
            Tensor
                The log likelihood. [B,1]
        """
        log_likelihood = 0
        for t in range(designs.shape[1]): #iterate through T steps
            intermediate_output = self.outcome_likelihood(log_k,alpha, designs[:,t,:]).log_prob(outcomes[:,t,:]) #[B,1] or [L+1,B,1] 
            log_likelihood += intermediate_output # [B,1] or [L+1,B,1]
        
            if torch.isnan(intermediate_output).any() or torch.isinf(intermediate_output).any():
                breakpoint()

        return log_likelihood
    
    @torch.no_grad()
    def estimate(self,log_k_0: Tensor,alpha_0: Tensor, designs: Tensor, outcomes: Tensor ):
        """
        Estimate the mutual information between params and outcomes (EIG)
        I(params; outcomes) = \E_{p(params)p(outcomes|params, designs)} [
            log p(outcomes|params, designs) - log p(outcomes|designs)
        ]
        
        log_k: Tensor of shape [B, 1]
        alpha: Tensor of shape [B, 1]
        designs: Tensor of shape [T, B, p]
        outcomes: Tensor of shape [T, B, 1]

        returns: float
        """

        L= self.num_inner_samples

        primary_log_likelihood = self.log_likelihood(log_k_0,alpha_0, designs, outcomes) #[B,1]


        
        
        log_k_l = self.model.prior_log_k_dist.sample((L, log_k_0.shape[0])) #[L,B,1]
        alpha_l = self.model.prior_alpha_dist.sample((L, alpha_0.shape[0])) #[L,B,1]

        if self.lower_bound:
                
            #concatenate the initial sample
            log_k_l = torch.cat([log_k_0.unsqueeze(0), log_k_l], dim=0) # [L+1,B,1]
            alpha_l = torch.cat([alpha_0.unsqueeze(0), alpha_l], dim=0) # [L+1,B,1]
           

        log_p_y_given_theta_l = self.log_likelihood(log_k_l,alpha_l, designs, outcomes) #[L+1,B,1]

        denominator = torch.logsumexp(log_p_y_given_theta_l, dim=0)  #[B,1]

        estimate = (primary_log_likelihood- denominator).mean(0) + math.log(L+self.lower_bound) #take the mean over the batch 

        return estimate.item()
    
    def differentiable_loss(self, log_k_0: Tensor,alpha_0: Tensor, designs: Tensor, outcomes: Tensor):
        """
        Args:
            log_k: Tensor of shape [B, 1]
            alpha: Tensor of shape [B, 1]
            designs: Tensor of shape [T, B, p]
            outcomes: Tensor of shape [T, B, 1]
        
        
        Returns:
            Tensor
                The differentiable loss. Tensor [1] 

        # implement the differentiable loss using reinforce
        """
            

        L= self.num_inner_samples

        primary_log_likelihood = self.log_likelihood(log_k_0,alpha_0, designs, outcomes) #[B,1]
        

        
        #no need to rsample as the prior distribution does not depend on params being optimised
        log_k_l = self.model.prior_log_k_dist.sample((L, log_k_0.shape[0])) #[L,B,1]
        alpha_l = self.model.prior_alpha_dist.sample((L, alpha_0.shape[0])) #[L,B,1]

        if self.lower_bound:
            #concatenate the initial sample
            log_k_l = torch.cat([log_k_0.unsqueeze(0), log_k_l], dim=0) # [L+1,B,1]
            alpha_l = torch.cat([alpha_0.unsqueeze(0), alpha_l], dim=0) # [L+1,B,1]
           

        log_p_y_given_theta_l = self.log_likelihood(log_k_l,alpha_l, designs, outcomes) #[L+1,B,1]

        denominator = torch.logsumexp(log_p_y_given_theta_l, dim=0)  #[B,1]



        
        #no need to add the l+1 term as it's a constant
        ## REINFORCE gradient estimator
        mi_estimate_no_grad = (primary_log_likelihood - denominator).detach()
        diff_loss = -(mi_estimate_no_grad * primary_log_likelihood - denominator).mean() #take the mean over the batch

        #if loss is NaN, pause for debugging
        if torch.isnan(diff_loss).any() or torch.isinf(diff_loss).any():
            breakpoint()

        assert diff_loss.requires_grad, "loss does not have gradients and thus soemthing has gone wrong"
        assert denominator.requires_grad , "denominator should have gradients but does not"
       
        return diff_loss #[1]
    
#####################################################################################################################################################
class EncoderNetwork(nn.Module):
    def __init__(
        self,
        design_dim,
        osbervation_dim,
        hidden_dim,
        encoding_dim,
        include_t,
        T,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.include_t = include_t
        self.T = T
        self.activation_layer = activation()
        self.design_dim = design_dim
        if include_t:
            input_dim = design_dim + 1
        else:
            input_dim = design_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer_0 = nn.Linear(hidden_dim, encoding_dim)
        self.output_layer_1 = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y, t):
        if self.include_t:
            t = xi.new_tensor(t) / self.T
            x = torch.cat([t.expand(*xi.shape[:-1],-1), xi], axis=-1)
        else:
            x = xi
        x = self.input_layer(x)
        x = self.activation_layer(x)
        x = self.middle(x)
        x_0 = self.output_layer_0(x)
        x_1 = self.output_layer_1(x)
        x = y * x_1 + (1.0 - y) * x_0
        return x


class EmitterNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.activation_layer = activation()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.output_dim = output_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, r):
        x = self.input_layer(r)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x



if __name__ == "__main__":

    #select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        breakpoint()
    #set the random seed
    torch.manual_seed(2)

    #set the batch size
    batch_size = 2000
    T= 20
    num_steps = 100000
    num_inner_samples = 5000
    num_outer_samples = 2000
    lr = 1e-4
    betas = (0.9, 0.999)
    gamma = 0.96
    annealing_frequency = 1000
    design_network_type = "static"
    design_dim = 2
    hidden_dim = 256
    encoding_dim = 16
    num_layers = 2
    include_t = False


    #####################################################################################################################################################
    class StaticDesignNetwork(nn.Module):
        def __init__(self, design_shape: int, T: int, designs: Tensor | None = None, num_past_datapoints: int = 0):
            super().__init__()
            self.design_shape = design_shape
            self.T = T
            if designs is None:
                designs = nn.Parameter(torch.rand(T, design_shape) * 2 - 1.0)
            else:
                designs = nn.Parameter(designs)
            self.register_parameter("designs", designs)

        def forward(self, designs: Tensor, outcomes: Tensor, num_past_datapoints: int=0) -> Tensor:
            t = designs.shape[1]-num_past_datapoints
            batch_size = designs.shape[0]
            return self.designs[t].expand(batch_size, self.design_shape)
    
    #####################################################################################################################################################
            
    encoder = EncoderNetwork(
        design_dim=design_dim,
        osbervation_dim=1,
        hidden_dim=hidden_dim,
        encoding_dim=encoding_dim,
        include_t=include_t,
        T=T,
        n_hidden_layers=num_layers,
    )
    emitter = EmitterNetwork(
        input_dim=encoding_dim,
        hidden_dim=hidden_dim,
        output_dim=design_dim,
        n_hidden_layers=num_layers,
    )

    if design_network_type == "static":
        design_net = StaticDesignNetwork(design_shape=2, T=T).to(device)
        lr = 1e-1
    elif design_network_type == "dad":
        #design_net = Dad(encoder, emitter, empty_value=torch.ones(design_dim)).to(device)
        design_net = DeepAdaptiveDesign( design_shape=2, T=T, y_dim=1, hidden_dim=128, embedding_dim=32)
        lr = 1e-4
    else:
        raise ValueError(f"design_network_type={design_network_type} not supported.")


    model = TemporalDiscounting(
        device=device,
        design_net=design_net,
        log_k_loc=-4.25,
        log_k_scale=1.5,
        prior_alpha= dist.Normal(torch.tensor([0.0],device=device), torch.tensor([2.0],device=device)),
        T=T,
        short_delay=0.0,
        long_reward=100.0,
        epsilon=0.01
    ).to(device)

    log_k, alpha, designs, outcomes = model(batch_size)
    assert log_k.shape == (batch_size, 1)
    assert alpha.shape == (batch_size, 1)
    assert designs.shape == (batch_size, T, 2)
    assert outcomes.shape == (batch_size, T, 1)


    objective = PriorContrastiveEstimation(
        model=model,
        outcome_likelihood=model.outcome_likelihood,
        num_inner_samples=num_inner_samples,
        lower_bound=True
    )

    objective_upper = PriorContrastiveEstimation(
        model=model,
        outcome_likelihood=model.outcome_likelihood,
        num_inner_samples=num_inner_samples,
        lower_bound=False
    )

    mlflow.set_experiment("Temporal Discounting")
    with mlflow.start_run() as run:
        #rename run to be more descriptive
        mlflow.set_tag("mlflow.runName", design_network_type)

        mlflow.log_params({
            "T": T,
            "num_steps": num_steps,
            "num_inner_samples": num_inner_samples,
            "num_outer_samples": num_outer_samples,
            "lr": lr,
            "betas": betas,
            "gamma": gamma,
            "annealing_frequency": annealing_frequency,
            "design_network_type": design_network_type,
            "batch_size": batch_size,
        })

                #estimate the initial mutual information averaged 
        mi_estimate_lower_list = []
        mi_estimate_upper_list = []
        for _ in trange(200):
            log_k, alpha, designs, outcomes = model(batch_size)
            mi_estimate_lower_list.append( objective.estimate(log_k, alpha, designs, outcomes))
            mi_estimate_upper_list.append(objective_upper.estimate(log_k, alpha, designs, outcomes))
        

        # Calculate the mean and standard deviation
        mi_estimate_lower = np.mean(mi_estimate_lower_list)
        mi_estimate_upper = np.mean(mi_estimate_upper_list)
        mi_stddev_lower = np.std(mi_estimate_lower_list)
        mi_stddev_upper = np.std(mi_estimate_upper_list)
        mi_estimate_lower = np.mean(mi_estimate_lower_list)


        mlflow.log_metric("mi_estimate_lower_init", mi_estimate_lower)
        mlflow.log_metric("mi_estimate_upper_init", mi_estimate_upper)

        mlflow.log_metric("mi_stddev_lower_init", mi_stddev_lower)
        mlflow.log_metric("mi_stddev_upper_init", mi_stddev_upper)

        print(f"Estimated MI lower_init: {mi_estimate_lower}  ± {mi_stddev_lower}")
        print(f"Estimated MI upper_init: {mi_estimate_upper} ± {mi_stddev_upper}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)

        for step in trange(num_steps):
            optimizer.zero_grad()
            log_k, alpha, designs, outcomes = model(batch_size)
            loss = objective.differentiable_loss(log_k, alpha, designs, outcomes)
            loss.backward()
            optimizer.step()
            

            if step % annealing_frequency == 0:
                scheduler.step()

            if step %10000 == 0:
                mi_estimate = objective.estimate(log_k, alpha, designs, outcomes)
                mlflow.log_metric("train_mi_estimate_lower", mi_estimate, step=step)
            mlflow.log_metric("loss", loss.item(), step=step)

        
        #estimate the final mutual information averaged 
        mi_estimate_lower_list = []
        mi_estimate_upper_list = []
        for _ in trange(num_outer_samples):
            log_k, alpha, designs, outcomes = model(batch_size)
            mi_estimate_lower_list.append( objective.estimate(log_k, alpha, designs, outcomes))
            mi_estimate_upper_list.append(objective_upper.estimate(log_k, alpha, designs, outcomes))
        

        # Calculate the mean and standard deviation
        mi_estimate_lower = np.mean(mi_estimate_lower_list)
        mi_estimate_upper = np.mean(mi_estimate_upper_list)
        mi_stddev_lower = np.std(mi_estimate_lower_list)
        mi_stddev_upper = np.std(mi_estimate_upper_list)
        mi_estimate_lower = np.mean(mi_estimate_lower_list)


        mlflow.log_metric("mi_estimate_lower", mi_estimate_lower)
        mlflow.log_metric("mi_estimate_upper", mi_estimate_upper)

        mlflow.log_metric("mi_stddev_lower", mi_stddev_lower)
        mlflow.log_metric("mi_stddev_upper", mi_stddev_upper)

        print(f"Estimated MI lower: {mi_estimate_lower}  ± {mi_stddev_lower}")
        print(f"Estimated MI upper: {mi_estimate_upper} ± {mi_stddev_upper}")