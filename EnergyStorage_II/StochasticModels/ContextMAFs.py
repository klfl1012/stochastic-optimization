import normflows as nf
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Any


class Context_Model(nn.Module):
    """
    Parent class for all context models. 
    Context models are used to model the conditional information of a given input to later combine with the output of the flow model.

    Child classes must implement the forward method.
    Child classes dont have a linear layer at the end, since the ouput of the context model is directly given as input to the context layer of the MADE network, which is a linear layer.
    """
    def __init__(self):
        super(Context_Model, self).__init__()

    def forward(self, x, context=None):
        raise NotImplementedError


class Context_RNN(Context_Model):
    """
    RNN model for processing the context.

    Parameters
    ----------
    :input_size: int - Dimension of the input data
    :hidden_size: int - Dimension of the hidden state
    :num_layers: int - Number of layers in the RNN
    :dropout: float - Dropout probability
    :output_size: int - Dimension of the output data
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, output_size: int):
        super(Context_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, context) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, context.size(0), self.hidden_size).to(context.device)
        out, _ = self.rnn(context, h0)
        return self.fc(out[:, -1, :])
    

class Context_MLP(Context_Model):
    """
    MLP model for processing the context. 

    Parameters
    ----------
    :input_size: int - Dimension of the input data
    :hidden_size: int - Dimension of the hidden state
    :num_layers: int - Number of layers in the MLP
    :output_size: int - Dimension of the output data
    :dropout: float - Dropout probability
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float):
        super(Context_MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        layers = [nn.Linear(input_size, hidden_size), nn.SiLU(), nn.Dropout(dropout)]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, context) -> torch.Tensor:
        if context.ndim == 3 and context.size(1) == 1:
            context = context.squeeze(1)

        return self.network(context)  
    

class Context_LSTM(Context_Model):
    """
    LSTM model for processing the context.

    Parameters
    ----------
    :input_size: int - Dimension of the input data
    :hidden_size: int - Dimension of the hidden state
    :num_layers: int - Number of layers in the LSTM
    :dropout: float - Dropout probability
    :output_size: int - Dimension of the output data    
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, output_size: int):
        super(Context_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, context) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, context.size(0), self.hidden_size).to(context.device)
        c0 = torch.zeros(self.num_layers, context.size(0), self.hidden_size).to(context.device)
        out, _ = self.lstm(context, (h0, c0))
        return self.fc(out[:, -1, :])
    
    
class Context_Transformer(Context_Model):
    """
    Transformer model for processing the context.

    Parameters
    ----------
    :param d_model: int - Dimension of the model
    :param nhead: int - Number of attention heads
    :param num_encoder_layers: int - Number of encoder layers
    :param dim_feedforward: int - Dimension of the feedforward network
    :param dropout: float - Dropout probability
    :param output_size: int - Dimension of the output data
    """

    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, dim_feedforward: int, dropout: float, output_size: int):
        super(Context_Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        encoder_layer.use_nested_tensor = False
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, output_size)


    def forward(self, context) -> torch.Tensor:
        transformer_output = self.transformer_encoder(context)
        out = transformer_output[:, -1, :]
        return self.fc(out)
    

class Conditional_MAF(nn.Module):
    """
    CNF model with an autoregressive model to process the context.

    Parameters
    ----------
    :x_dim: int - Dimension of the input data
    :c_dim: int - Dimension of the context data
    :q0: Distribution - Initial distribution for the flow
    :context_model: ARModel - Autoregressive model for processing the context
    :num_flows: int - Number of normalizing flows
    :hidden_features: int - Number of hidden features in the flow
    :num_blocks: int - Number of blocks in the flow
    :dropout_probability: float - Dropout probability in the flow
    :use_residual_blocks: bool - Whether to use residual blocks in the flow
    :use_batch_norm: bool - Whether to use batch normalization in the flow
    :lr: float - Learning rate for the optimizer
    :temperature: float - Temperature for the loss function
    :weight_decay: float - Weight decay for the optimizer
    :p: target - target distribution -> for training on reverse KL divergence (not implemented)
    """
    def __init__(
            self, 
            x_dim: int, 
            c_dim: int, 
            context_model: Union[Context_Model, nn.Module], 
            trainable_q0: bool, 
            num_flows: int, 
            hidden_features: int, 
            num_blocks: int, 
            dropout_probability: float, 
            use_residual_blocks: bool, 
            use_batch_norm: bool, 
            temperature: float, 
            weight_decay: float, 
            lr: float=0.001, 
            p: Any=None
        ):
        super(Conditional_MAF, self).__init__()
        assert context_model is not None, "context_model must be provided"
        assert isinstance(context_model, (Context_Model, nn.Module)), "context_model must be a instance of Context_Model or nn.Module"

        self.context_model = context_model
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.trainable_q0 = trainable_q0
        self.num_flows = num_flows
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.dropout_probability = dropout_probability
        self.use_residual_blocks = use_residual_blocks
        self.use_batch_norm = use_batch_norm
        self.lr = lr
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.p = p

        self.cnf = nf.ConditionalNormalizingFlow(
            q0=nf.distributions.DiagGaussian(x_dim, trainable=trainable_q0),
            flows=nn.ModuleList(
                [nf.flows.MaskedAffineAutoregressive(
                    features=x_dim,
                    context_features=c_dim,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks,
                    dropout_probability=dropout_probability,
                    use_residual_blocks=use_residual_blocks,
                    use_batch_norm=use_batch_norm
                ) for _ in range(num_flows)]
            ),
            p=p
        )
        self.optimizer = torch.optim.Adam(list(set(self.context_model.parameters()).union(set(self.parameters()))), lr=lr, weight_decay=weight_decay)

    def forward_kld(self, x, context=None):
        """
        Forward pass the context threw the context model and calculate the forwad KL divergence loss.
        """
        c_out = self.context_model(context)
        return self.cnf.forward_kld(x, context=c_out)

    def fit(self, X_train: torch.Tensor, c_train: torch.Tensor, num_epochs: int=100, relative_improvement_threshold: float=1e-2, early_stopping_patience: int=5, verbose: bool=True):
        """
        Train the model on the given data.

        Parameters
        ----------
        :X_train: torch.Tensor - Training data
        :c_train: torch.Tensor - Training context
        :num_epochs: int - Number of epochs
        :relative_improvement_threshold: float - Threshold for early stopping
        :early_stopping_patience: int - Patience for early stopping
        :verbose: bool - Whether to print training information
        """
        self.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            self.optimizer.zero_grad(set_to_none=True)

            loss = self.forward_kld(X_train, context=c_train) * self.temperature

            if loss < 0 or torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f"Training diverged at epoch {epoch} with loss {loss}")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(set(self.context_model.parameters()).union(set(self.parameters()))), 1.0)
            self.optimizer.step()

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss {loss}")
            
            if abs(best_loss - loss.item()) > relative_improvement_threshold:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} with best loss {best_loss}")
                break
        
        if verbose:
            print(f"Final training loss: {best_loss}")

    def generate_sample_paths(self, num_sample_paths: int, num_samples: int, context: torch.Tensor, noise: float=0.0) -> np.ndarray:
        """
        Generate sample paths by sampling from the model.

        Parameters
        ----------
        :num_sample_paths: int - Number of sample paths
        :num_samples: int - Number of samples per path
        :context: torch.Tensor - Context for the samples
        :noise: float - Noise for the samples

        Returns
        -------
        :np.ndarray - Array of sample
        """
        self.eval()
        sample_paths = []
        with torch.no_grad():
            for _ in range(num_sample_paths):
                c_out = self.context_model(context)
                samples, _ = self.cnf.sample(num_samples, context=c_out)
                if noise > 0:
                    samples += torch.randn_like(samples) * noise
                sample_paths.append(samples)
        return np.array(sample_paths)

    def save(self, path):
        """
        Saves the model to the given path.
        """
        self.eval()
        if self.context_model.__class__.__name__ in ["Context_MLP", "Context_RNN", "Context_LSTM"]:
            context_model_params = {
                "input_size": self.context_model.input_size,
                "hidden_size": self.context_model.hidden_size,
                "num_layers": self.context_model.num_layers,
                "dropout": self.context_model.dropout,
                "output_size": self.context_model.output_size
            }
        elif self.context_model.__class__.__name__ == "Context_Transformer":
            context_model_params = {
                "d_model": self.context_model.d_model,
                "nhead": self.context_model.nhead,
                "num_encoder_layers": self.context_model.num_encoder_layers,
                "dim_feedforward": self.context_model.dim_feedforward,
                "dropout": self.context_model.dropout,
                "output_size": self.context_model.output_size
            }
        else:
            raise ValueError(f"Unknown class for context_model: {self.context_model.__class__.__name__}")
        
        torch.save({
            "context_model_class": self.context_model.__class__.__name__,
            "context_model_state_dict": self.context_model.state_dict(),
            "context_model_params": context_model_params,
            "cnf_state_dict": self.cnf.state_dict(),
            "cnf_params": {
                "x_dim": self.x_dim,
                "c_dim": self.c_dim,
                "num_flows": self.num_flows,
                "hidden_features": self.hidden_features,
                "num_blocks": self.num_blocks,
                "dropout_probability": self.dropout_probability,
                "use_residual_blocks": self.use_residual_blocks,
                "use_batch_norm": self.use_batch_norm,
                "q0_trainable": self.trainable_q0,
                "p": self.p,
            },
        }, path)

    def load(self, path):
        """
        Loads the model from the given path.
        """
        context_model_classes = {
            "Context_MLP": Context_MLP,
            "Context_RNN": Context_RNN,
            "Context_LSTM": Context_LSTM,
            "Context_Transformer": Context_Transformer
        }

        checkpoint = torch.load(path, weights_only=False)
        context_model_class_name = checkpoint["context_model_class"]
        context_model_class = context_model_classes[context_model_class_name]
        context_model_params = checkpoint["context_model_params"]
        relevant_params = {k: v for k, v in context_model_params.items() if k in context_model_classes[context_model_class_name].__init__.__code__.co_varnames}
        self.context_model = context_model_class(**relevant_params)
        self.context_model.load_state_dict(checkpoint["context_model_state_dict"])

        cnf_params = checkpoint["cnf_params"]
        self.cnf = nf.ConditionalNormalizingFlow(
            q0=nf.distributions.DiagGaussian(cnf_params["x_dim"], trainable=cnf_params["q0_trainable"]),
            flows=nn.ModuleList([
                nf.flows.MaskedAffineAutoregressive(
                    features=cnf_params["x_dim"],
                    context_features=cnf_params["c_dim"],
                    hidden_features=cnf_params["hidden_features"],
                    num_blocks=cnf_params["num_blocks"],
                    dropout_probability=cnf_params["dropout_probability"],
                    use_residual_blocks=cnf_params["use_residual_blocks"],
                    use_batch_norm=cnf_params["use_batch_norm"]
                ) for _ in range(cnf_params["num_flows"])
            ]),
            p=cnf_params["p"]
        )
        self.cnf.load_state_dict(checkpoint["cnf_state_dict"])
        self.eval()

