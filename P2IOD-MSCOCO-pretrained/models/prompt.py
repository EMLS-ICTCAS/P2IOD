import torch
import torch.nn as nn
import pdb
import copy
import torch.nn.functional as F

class PromptMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
        prompt_len: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.prompt_len = prompt_len

        non_linearity = nn.ReLU(inplace=True)
        if activation == "sigmoid":
            non_linearity = nn.Sigmoid()
        elif activation == "attention":
            non_linearity = nn.Softmax(dim=-1)

        self.block = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features, bias=bias),
            non_linearity,
            nn.Linear(self.hidden_features, self.out_features * self.prompt_len, bias=bias),
        )
        if dropout > 0.0:
            self.block[1].register_forward_hook(
                lambda m, inp, out: F.dropout(out, p=dropout, training=m.training)
            )

    def forward(self, x: torch.Tensor):
        bsz = x.size(0)
        out = self.block(x)
        out = out.reshape(bsz, self.prompt_len, self.out_features)

        i = int(self.prompt_len/2)
        Ek = out[:,:i,:]
        Ev = out[:,i:,:]

        p_return = [Ek, Ev]
        return p_return
    
    def l1_regularization(self, lambda_l1_norm: float = 1.0):
        l1_norm = 0.0
        for param in self.parameters():
            l1_norm += param.abs().sum()  
        return lambda_l1_norm * l1_norm


class ParameterizedPrompt(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        use_bias: bool = False,
        embed_dim: int = 256,
        dropout: float = 0.0,
        activation: str = "relu",
        args=None
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = args.mlp_hidden_dim
        self.use_bias = use_bias
        self.prompt_len = args.prompt_len
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.activation = activation
        self.args = args


        self.init()

    def init(self):
        self.memory_list = nn.ModuleList()
        out_dim = self.embed_dim
        in_dim = self.embed_dim

        if self.args.sparse_prompt:
            self.l1_norm_losses = 0.0

        for _ in range(self.num_layers):
            module = PromptMLP(
                in_dim,
                out_dim,
                self.hidden_dim,
                self.use_bias,
                self.dropout,
                self.prompt_len,
                self.activation,
            )
            self.memory_list.append(module)

    def forward(self, x_querry: torch.Tensor):
        outputs = list()
        if self.args.sparse_prompt:
            l1_norm_losses=0.0
        for i, module in enumerate(self.memory_list):
            x = x_querry  # bsz, ndim
            if self.args.sparse_prompt:
                l1_norm_losses += module.l1_regularization(self.args.lambda_l1_norm)
            outputs.append(module(x))
        if self.args.sparse_prompt:
            self.l1_norm_losses = l1_norm_losses

        return outputs
    
    def get_prompt_memory_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for param in list(self.memory_list.parameters()):
            params.append(param.view(-1).data.clone())
        return params
    
    def set_memory_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        for pp, new_param in zip(self.memory_list.parameters(), new_params):
            new_param = new_param.view(pp.size())
            pp.data = new_param.clone()
    
