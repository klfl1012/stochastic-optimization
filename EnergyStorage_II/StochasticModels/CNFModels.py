import normflows as nf
import torch
import torch.nn as nn
import numpy as np


class CNF(nf.ConditionalNormalizingFlow):
    """
    Conditional Normalizing Flow model for generating sample scenarios of given data.
    The CNF is constructed using a DiagGaussian as a base ditribution and a MaskedAffineAutoregressive flow. 
    The model can be trained using the forward Kulback-Leibler divergence between the data and the model.

    Parameters
    ----------
    :x_dim: int - Dimensionality of the input data
    :c_dim: int - Dimensionality of the context data
    :trainable_q0: bool - Whether the base distribution is trainable or not
    :num_flows: int - Number of flows in the model
    :hidden_features: int - Number of hidden features in the model
    :num_blocks: int - Number of blocks in the model
    :dropout_probability: float - Dropout probability in the model
    :use_batch_norm: bool - Whether to use batch normalization in the model
    :use_residual_blocks: bool - Whether to use residual blocks in the model
    :weight_decay: float - Weight decay for the optimizer
    :temperature: float - Temperature for the model
    :lr: float - Learning rate for the optimizer
    """
    def __init__(
            self, 
            x_dim: int, 
            c_dim: int, 
            trainable_q0: bool, 
            num_flows: int, 
            hidden_features: int, 
            num_blocks: int, 
            dropout_probability: float, 
            lr: float, 
            weight_decay: float, 
            temperature: float, 
            use_batch_norm: bool, 
            use_residual_blocks: bool
        ):
        super().__init__(None, None)
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.trainable_q0 = trainable_q0
        self.num_flows = num_flows
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.dropout_probability = dropout_probability
        self.use_batch_norm = use_batch_norm
        self.use_residual_blocks = use_residual_blocks
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.lr = lr

        self.base_dist = nf.distributions.DiagGaussian(x_dim, trainable=trainable_q0)
        self.flows = nn.ModuleList([
            nf.flows.MaskedAffineAutoregressive(
                features= x_dim,
                context_features=c_dim,
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
                use_residual_blocks=use_residual_blocks
            ) for _ in range(num_flows)
        ])
        super().__init__(self.base_dist, self.flows)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward_kld(self, x, context=None):
        """
        Computes the forward Kulback-Leibler divergence between the data and the model. 
        """
        return super().forward_kld(x, context)

    def fit(self, X_train: torch.Tensor, c_train: torch.Tensor, num_epochs: int=100, relative_improvement_threshold: float=1e-2, early_stopping_patience: int=5, verbose: bool=True):
        """
        Train the model on the given data.

        Parameters
        ----------
        :X_train: torch.Tensor - Training data
        :c_train: torch.Tensor - Training context
        :num_epochs: int - Number of epochs to train the model
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
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
        Generate sample paths from the model.

        Parameters
        ----------
        :num_sample_paths: int - Number of sample paths to generate
        :num_samples: int - Number of samples in each path
        :context: torch.Tensor - Context for the samples
        :noise: float - Noise to add to the samples

        Returns
        -------
        :np.ndarray: Sample paths
        """
        super().eval()
        sample_paths = []
        with torch.no_grad():
            for _ in range(num_sample_paths):
                samples, _ = self.sample(num_samples, context=context)
                if noise > 0:
                    samples += torch.randn_like(samples) * noise
                sample_paths.append(samples)
        return np.array(sample_paths)
    
    # def generate_sample_paths(self, num_sample_paths: int, num_samples: int, context: torch.Tensor, noise: float = 0.0, mean_over: int = 1) -> np.ndarray:
    #     """
    #     Generate sample paths from the model and optionally compute the mean over multiple paths.

    #     Parameters
    #     ----------
    #     :num_sample_paths: int - Total number of sample paths to generate
    #     :num_samples: int - Number of samples in each path
    #     :context: torch.Tensor - Context for the samples
    #     :noise: float - Noise to add to the samples
    #     :mean_over: int - Number of paths to average over (default: 1, meaning no averaging)

    #     Returns
    #     -------
    #     :np.ndarray: Sample paths (either raw or averaged)
    #     """
    #     if mean_over < 1 or mean_over > num_sample_paths:
    #         raise ValueError("mean_over must be between 1 and num_sample_paths")

    #     super().eval()
    #     sample_paths = []
        
    #     with torch.no_grad():
    #         for _ in range(num_sample_paths):
    #             samples, _ = self.sample(num_samples, context=context)
    #             if noise > 0:
    #                 samples += torch.randn_like(samples) * noise
    #             sample_paths.append(samples.numpy())  # Konvertiere zu numpy für einfachere Verarbeitung

    #     sample_paths = np.array(sample_paths)  # (num_sample_paths, num_samples, ...)

    #     # Falls mean_over > 1, mitteln wir über Gruppen von mean_over Pfaden
    #     if mean_over > 1:
    #         num_aggregated_paths = num_sample_paths // mean_over  # Wie viele gemittelte Pfade es gibt
    #         sample_paths = sample_paths[:num_aggregated_paths * mean_over]  # Schneide überzählige Pfade ab
    #         sample_paths = sample_paths.reshape(num_aggregated_paths, mean_over, num_samples, -1)  # Gruppiere
    #         sample_paths = np.mean(sample_paths, axis=1)  # Mittele über die Gruppen

    #     return sample_paths  # (num_aggregated_paths, num_samples, ...)


    def save(self, path):
        """
        Save the model to the given path.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load the model from the given path.
        """
        self.load_state_dict(torch.load(path, weights_only=True))
