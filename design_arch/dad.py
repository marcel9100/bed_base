import torch
from torch import Tensor

from torch import nn
from tqdm import trange

class EncoderNetwork(nn.Module):
    """Encoder network for location finding example"""

    def __init__(self, design_dim, osbervation_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim
        #self.design_dim_flat = design_dim[0] * design_dim[1]
        input_dim = design_dim[1] + osbervation_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, xi, y, **kwargs):
        """
        Forward pass of the model.
        Args:
            xi: 
                [batch_size, design_dim]
            y: 
                [batch_size, observation_dim]
        """
        #xi = xi.flatten(-2)
        inputs = torch.cat([xi, y], dim=-1) #[b, design_dim+observation_dim]

        x = self.linear1(inputs) #[b, hidden_dim]
        x = self.relu(x) #[b, hidden_dim]
        x = self.output_layer(x) #[b, encoding_dim]
        return x


class EmitterNetwork(nn.Module):
    """Emitter network for location finding example"""

    def __init__(self, encoding_dim, design_dim):
        super().__init__()
        self.design_dim = design_dim
        #self.design_dim_flat = design_dim[0] * design_dim[1]
        self.linear = nn.Linear(encoding_dim, design_dim[1])

    def forward(self, r):
        xi_flat = self.linear(r) 
        return xi_flat


class Dad(torch.nn.Module):
    def __init__(
        self,
        encoder_network,
        emission_network,
        empty_value,
    ):
        super().__init__()
        self.encoder = encoder_network
        self.emitter = emission_network
        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))


    def forward(self, designs: Tensor, outcomes: Tensor, num_past_datapoints) -> Tensor:
        """
        Forward pass of the model.
        Args:
            designs: 
                [batch_size,t+num_past_datapoints, design_dim]
            outcomes: 
                [batch_size,t+num_past_datapoints, observation_dim]
            num_past_datapoints: 
                int, number of past datapoints to consider for the model
        """
        
        #update to ensure it is searching for the right dim
        # [TODO]
        if designs.shape[1] == 0:
            #new zeros used to inherit the device of the empty_value
            sum_encoding = self.empty_value.new_zeros(designs.shape[0],self.encoder.encoding_dim)

        else:
            sum_encoding = sum(
                self.encoder(xi=designs[:,idx,:], y=outcomes[:,idx,:], t=[idx + 1])
                for idx in range(designs.shape[1]) 
            ) #[b, encoding_dim]
        output = self.emitter(sum_encoding) #[b, design_dim]

        return output


############################################
##desi implementation
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
        t = designs.shape[1]
        batch_size = designs.shape[0]

        # Encode designs and outcomes
        encoded = self.encode_designs(designs)
        x_0 = self.head0(encoded)
        x_1 = self.head1(encoded)
        x = outcomes * x_1 + (1.0 - outcomes) * x_0

        # Sum over time steps
        x = x.sum(dim=1)  # [B, embedding_dim]
        if self.time_embedding:
            # self.time_embeddings[t] is [embedding_dim]
            time_embedding = self.time_embeddings[t].expand(*x.shape)
            # concat time embedding with the summed features
            x = torch.cat([x, time_embedding], dim=-1)
            x = self.time_projection(x)

        return self.decode_designs(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


    batch_size = 10
    p=2
    T=10
    design_dim = (batch_size, p)
    observation_dim = 1
    hidden_dim = 256
    encoding_dim = 16
    empty_value = torch.zeros(batch_size,0,encoding_dim)
    encoder = EncoderNetwork(design_dim, observation_dim, hidden_dim, encoding_dim)
    emitter = EmitterNetwork(encoding_dim, design_dim)
    design_net = Dad(encoder, emitter, empty_value).to(device)
    designs = torch.zeros(batch_size, 0, p, device=device)
    outcomes = torch.zeros(batch_size, 0, observation_dim, device=device)
    num_past_datapoints = 0

    for t in trange(T):
        xi = design_net(designs, outcomes, num_past_datapoints).to(device) # [batch_size, design_dim]
        yi = torch.randn(batch_size, 1, observation_dim, device=device) # [batch_size, 1,observation_dim]
        designs = torch.cat([designs, xi.unsqueeze(1)], dim=1) # [batch_size, t+1, design_dim]
        outcomes = torch.cat([outcomes, yi], dim=1) # [batch_size, t+1, observation_dim]
        print(designs.shape)

    print("=====")
    print(designs.shape)
    print(outcomes.shape)  
