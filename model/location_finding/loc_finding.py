import torch
import torch.nn as nn
import torch.distributions as dist

from torch import Tensor
import math

from typing import Callable

import mlflow

from design_arch.dad import EncoderNetwork, EmitterNetwork, Dad

from tqdm import trange

import numpy as np

#print the system path
import sys
print(sys.path)
print("=====================================================================================================")

class HiddenObjects(nn.Module):
    """
    Implementing location finding as in https://github.com/ae-foster/dad/blob/master/dad/loc_finding.py
    """

    def __init__(self,
             design_net, # design network
             device, # device to run on
             base_signal=0.1,  #  hyperparam
             max_signal=1e-4,  # hyperparam
             theta_loc=None,  # prior on theta mean hyperparam
             theta_covmat=None,  # prior on theta covariance hyperparam
             noise_scale=None,  # this is the scale of the noise term
             p=1,  # physical dimension
             K=2,  # number of sources
             T=10,  # number of experiments
             ):
        
        super().__init__()
        self.design_net = design_net
        self.base_signal = base_signal
        self.max_signal = max_signal
        self.theta_loc = theta_loc #[K, p]
        self.theta_covmat = theta_covmat # [p,p]
        #assert that theta loc and theta covmat are of the right shape
        assert self.theta_loc.shape[0] == K
        assert self.theta_loc.shape[1] == p
        assert self.theta_covmat.shape[0] == p
        assert self.theta_covmat.shape[1] == p

        self.noise_scale = noise_scale
        self.p = p
        self.K = K
        self.T = T
        self.device = device

        #now to set up the prior
        #batching handled during sampling
        self.theta_prior = dist.MultivariateNormal(loc=theta_loc, covariance_matrix=theta_covmat)
    
    def outcome_likelihood(
        self,
        theta, # [B, K, p]
        design, # [B, p]
        )-> dist.Distribution:
        """
        Implementing the likelihood model

        Args:
            theta: [B, K, p]
            design: [B, p]

        
        Returns:
            dist.Distribution
                The distribution of the outcome.
        """
        # two norm squared
        sq_two_norm = (design.unsqueeze(1) - theta).pow(2).sum(axis=-1) # [B, K] - we sum over the p dimension to get the two norm squared
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1) # [B, K] - add max signal as in the paper likely so never dividing by zero
        # sum over the K sources, add base signal and take log.
        #note as in the paper, the alpha is one hence why not included in the implementation
        mean_y = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True)) # [B, 1]
        
        #the input to the normal distribution is the mean and the scale with dimensions [B, 1] so the samples will be [B, 1] independent samples
        y_dist = dist.Normal(mean_y, self.noise_scale) #note that y is a scalar hence univariate normal

        return y_dist

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
                The design matrix of shape [batch_size,t, p]
            past_outcomes: torch.Tensor
                The outcome matrix of shape [batch_size,t, 1]

        Returns:
            theta: torch.Tensor
                The estimated theta of shape [batch_size, K, p]
            design: torch.Tensor
                The design matrix of shape [batch_size,t+num_past_datapoints, p]
            outcome: torch.Tensor
                The outcome matrix of shape [batch_size, t+num_past_datapoints,1]
        """
        
        #sample theta from the prior
        theta = self.theta_prior.sample([batch_size]).to(self.device) # [B, K, p]

        if past_designs is None:
            num_past_datapoints = 0
        else:
            num_past_datapoints = past_designs.shape[1]
            assert past_outcomes.shape[:-1] == past_designs.shape[:-1] #same shape up till the last dimension

        #set up empty design and outcome tensors
        designs = torch.zeros(batch_size, 0, self.p, device=self.device) if past_designs is None else past_designs
        outcomes = torch.zeros(batch_size, 0, 1, device=self.device) if past_outcomes is None else past_outcomes

        for t in range(num_past_datapoints, self.T+num_past_datapoints):
            xi = self.design_net(designs, outcomes, num_past_datapoints).to(self.device) # [B, p]

            #sample the outcome
            yi = self.outcome_likelihood(theta, xi).rsample() # [B, 1] # rsample is used to get the samples but also keep the gradient

            #append the design and outcome to the list
            designs = torch.cat([designs, xi.unsqueeze(1)], dim=1) # [B, t+num_past_datapoints, p]
            outcomes = torch.cat([outcomes, yi.unsqueeze(1)], dim=1) # [B, t+num_past_datapoints, 1]

            #print(f"Designs: {designs.shape}, Outcomes: {outcomes.shape}")
        #return the final design and outcome
        return theta, designs, outcomes
    
    @torch.no_grad()
    def run_policy(
            self,
            theta: Tensor,
            past_designs: Tensor | None = None,
            past_outcomes: Tensor | None = None,
            ) -> tuple[Tensor, Tensor]:
            """
            Run a trajectory of T designs for the given theta parameter
            """
            batch_size = theta.shape[0]

            if past_designs is None:
                num_past_datapoints = 0
            else:
                num_past_datapoints = past_designs.shape[1]
                assert past_outcomes.shape[:-1] == past_designs.shape[:-1] #same shape up till the last dimension

            #set up empty design and outcome tensors
            designs = torch.zeros(batch_size, 0, self.p, device=self.device) if past_designs is None else past_designs
            outcomes = torch.zeros(batch_size, 0, 1, device=self.device) if past_outcomes is None else past_outcomes


            
            for t in range(num_past_datapoints, self.T+num_past_datapoints):
                
                xi = self.design_net(designs, outcomes,num_past_datapoints).to(self.device) # [B, p]

                #sample the outcome
                yi = self.outcome_likelihood(theta, xi).sample() # [B, 1]

                #append the design and outcome to the list
                designs = torch.cat([designs, xi.unsqueeze(1)], dim=1) # [B, t+num_past_datapoints, p]
                outcomes = torch.cat([outcomes, yi.unsqueeze(1)], dim=1) # [B, t+num_past_datapoints, 1]

        
            return designs, outcomes # [B, T, p], [B, T, 1]


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
        
    def log_likelihood(self, theta, designs, outcomes):
        """
        Args:
            theta: Tensor
                The theta parameter. [B,K,p] or [L+1,B,K,p]
            designs: Tensor
                The designs. [B, T, p]
            outcomes: Tensor
                The outcomes. [B, T, 1]
        
        Returns:
            Tensor
                The log likelihood. [B,1]
        """
        log_likelihood = 0
        for t in range(designs.shape[1]): #iterate through T steps
            intermediate_output = self.outcome_likelihood(theta, designs[:,t,:]).log_prob(outcomes[:,t,:]) #[B,1] or [L+1,B,1] 
            log_likelihood += intermediate_output # [B,1] or [L+1,B,1]
        
            if torch.isnan(intermediate_output).any() or torch.isinf(intermediate_output).any():
                breakpoint()

        return log_likelihood
    
    @torch.no_grad()
    def estimate(self,theta_0: Tensor, designs: Tensor, outcomes: Tensor ):
        """
        Estimate the mutual information between params and outcomes (EIG)
        I(params; outcomes) = \E_{p(params)p(outcomes|params, designs)} [
            log p(outcomes|params, designs) - log p(outcomes|designs)
        ]
        
        theta: Tensor of shape [B, K, p]
        designs: Tensor of shape [T, B, p]
        outcomes: Tensor of shape [T, B, 1]

        returns: float
        """

        L= self.num_inner_samples

        primary_log_likelihood = self.log_likelihood(theta_0, designs, outcomes) #[B,1]


        
        
        theta_l = self.model.theta_prior.sample((L, theta_0.shape[0])) #[L,B,K,P]

        if self.lower_bound:
                
            #concatenate the initial sample
            theta_l = torch.cat([theta_0.unsqueeze(0), theta_l], dim=0) # [L+1,B,k,p]
           

        log_p_y_given_theta_l = self.log_likelihood(theta_l, designs, outcomes) #[L+1,B,1]

        denominator = torch.logsumexp(log_p_y_given_theta_l, dim=0)  #[B,1]

        estimate = (primary_log_likelihood- denominator).mean(0) + math.log(L+self.lower_bound) #take the mean over the batch 

        return estimate.item()
    
    def differentiable_loss(self, theta_0: Tensor, designs: Tensor, outcomes: Tensor):
        """
        Args:
            theta: Tensor
                The theta parameter. [B,K,p]
            designs: Tensor
                The designs. [T, B, p]
            outcomes: Tensor
                The outcomes. [T, B, 1]
        
        Returns:
            Tensor
                The differentiable loss. Tensor [1] 

        # implement the differentiable loss using rsample
        """
        #confirm that the outcomes have gradients and thus were chosen using rsample
        #and print "rsample likely not used" if the error is hit
        assert outcomes.requires_grad, "Outcomes do not have gradients and thus rsample likely not used"
            

        L= self.num_inner_samples

        primary_log_likelihood = self.log_likelihood(theta_0, designs, outcomes) #[B,1]
        

        
        #no need to rsample as the prior distribution does not depend on params being optimised
        theta_l = self.model.theta_prior.sample((L, theta_0.shape[0])) #[L,B,K,p]
        
        if self.lower_bound:
            #concatenate the initial sample
            theta_l = torch.cat([theta_0.unsqueeze(0),theta_l], dim=0) # [L+1,B,K,p]


        log_p_y_given_theta_l = self.log_likelihood(theta_l, designs, outcomes) #[L+1,B,1]

        denominator = torch.logsumexp(log_p_y_given_theta_l, dim=0)  #[B,1]

        #no need to add the l+1 term as it's a constant
        loss = -(primary_log_likelihood- denominator).mean()  #take the mean over the batch 
      

        #if loss is NaN, pause for debugging
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            breakpoint()
        return loss #[1]


if __name__ == "__main__":

    #select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        breakpoint()
    #set the random seed
    torch.manual_seed(0)

    #set the batch size
    batch_size = 1024
    T= 10
    p=2
    K=1
    theta_prior_loc = torch.zeros((K, p), device=device)  # mean of the prior
    theta_prior_covmat = torch.eye(p, device=device)  # covariance of the prior
    noise_scale_tensor = 0.5 * torch.tensor(
        1.0, dtype=torch.float32, device=device
        )
    adam_betas_wd=[0.8, 0.998, 0]
    *adam_betas, adam_weight_decay = adam_betas_wd
    hidden_dim = 256
    encoding_dim = 16
    num_steps = 50000
    num_inner_samples = 2000
    num_outer_samples = 2000
    design_network_type = "dad"
    
    #####################################################################################################################################################
    class RandomDesign(nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p
        def forward(self, designs, outcomes, num_past_datapoints):
            return torch.distributions.Normal(torch.zeros(self.p),torch.ones(self.p)).sample([designs.shape[0]])

    #####################################################################################################################################################
    class StaticDesignNetwork(nn.Module):
        """
        A static design network that returns that returns a random design at each time step
        """
        def __init__(self, design_shape: torch.Size, T: int, designs: Tensor | None = None):
            super().__init__()
            self.design_shape = design_shape
            self.T = T
            if designs is None:
                xi_init = torch.zeros(T,design_shape)
                designs = nn.Parameter(xi_init)
            else:
                designs = nn.Parameter(designs)
                if designs.dim() != 2:
                    raise ValueError("Passed designs should have shape [T, design_shape]")
            self.register_parameter("designs", designs)

        def forward(self, designs: Tensor, outcomes: Tensor, num_past_datapoints) -> Tensor:
            t = designs.shape[1]-num_past_datapoints
            batch_size = designs.shape[0]
            return self.designs[t].expand(batch_size, self.design_shape)
    
    n=batch_size #batch_dim
    encoder = EncoderNetwork(design_dim=(n,p), osbervation_dim=1, hidden_dim=hidden_dim, encoding_dim=encoding_dim)
    emitter = EmitterNetwork(encoding_dim=encoding_dim, design_dim=(n,p))
    

    if design_network_type == "static":
        design_net = StaticDesignNetwork(p,T).to(device)
    elif design_network_type == "random":
        design_net =  RandomDesign(p).to(device)
        num_steps = 0  # no gradient steps needed
    elif design_network_type == "dad":
        design_net = Dad(encoder, emitter, empty_value=torch.ones(n, p) * 0.01).to(device)
    else:
        raise ValueError(f"design_network_type={design_network_type} not supported.")


   

    #set up the model
    model = HiddenObjects(
        design_net = design_net,
        device = device,
        theta_loc=theta_prior_loc,
        theta_covmat=theta_prior_covmat,
        noise_scale=noise_scale_tensor,
        p=p,
        T=T,
        K=K,
    )


    #now set up a training loop
    objective = PriorContrastiveEstimation(
        model=model,
        outcome_likelihood=model.outcome_likelihood,
        num_inner_samples=num_inner_samples,
        lower_bound=True,
    )

    objective_upper = PriorContrastiveEstimation(
        model=model,
        outcome_likelihood=model.outcome_likelihood,
        num_inner_samples=num_inner_samples,
        lower_bound=False
    )    

    mlflow.set_experiment(f"loc_finding_model_04_11")
    with mlflow.start_run() as run:
        #set the run name
        mlflow.set_tag("mlflow.runName", design_network_type)


        mlflow.log_param("design_network_type", design_network_type)
        mlflow.log_param("T", T)
        mlflow.log_param("p", p)
        mlflow.log_param("K", K)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("encoding_dim", encoding_dim)
        mlflow.log_param("num_steps", num_steps)
        mlflow.log_param("noise_scale", noise_scale_tensor.item())
        mlflow.log_param("theta_prior_loc", theta_prior_loc.tolist())
        mlflow.log_param("theta_prior_covmat", theta_prior_covmat.tolist())
        mlflow.log_param("adam_betas", adam_betas)
        mlflow.log_param("adam_weight_decay", adam_weight_decay)
        mlflow.log_param("num_inner_samples", 2000)
        mlflow.log_param("lower_bound", True)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_inner_samples", num_inner_samples)
        mlflow.log_param("num_outer_samples", num_outer_samples)

        
        #estimate the initial mutual information averaged 
        mi_estimate_lower_list = []
        mi_estimate_upper_list = []
        for _ in trange(200):
            theta, designs, outcomes = model(batch_size)
            mi_estimate_lower_list.append( objective.estimate(theta, designs, outcomes))
            mi_estimate_upper_list.append(objective_upper.estimate(theta, designs, outcomes))
        

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

        if design_network_type != "random":

            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=adam_betas, weight_decay=adam_weight_decay)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            
            for step in trange(num_steps):
                optimizer.zero_grad()
                theta, designs, outcomes = model(batch_size)
                loss = objective.differentiable_loss(theta, designs, outcomes)
                loss.backward()
                optimizer.step()
                if step % 1000 == 0:
                    scheduler.step()
                    mi_estimate = objective.estimate(theta, designs, outcomes)
                    mlflow.log_metric("train_mi_estimate_lower", mi_estimate, step=step)
                #print(f"Loss: {loss.item()}")
                mlflow.log_metric("Loss", loss.item(),step=step)

        
        #estimate the final mutual information averaged 
        mi_estimate_lower_list = []
        mi_estimate_upper_list = []
        for _ in trange(num_outer_samples):
            theta, designs, outcomes = model(batch_size)
            mi_estimate_lower_list.append( objective.estimate(theta, designs, outcomes))
            mi_estimate_upper_list.append(objective_upper.estimate(theta, designs, outcomes))       

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