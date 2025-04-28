from abc import ABC, abstractmethod
import joblib
import yaml
import torch
import torch.nn
import numpy as np
import joblib
import sys
sys.path.append("../")

from StochasticModels.old.CNFWithAR import CNFWithARModel, Context_LSTM as Context_LSTM_old
from StochasticModels.ContextMAFs import Context_MLP, Context_RNN, Context_LSTM, Context_Transformer, Conditional_MAF
from StochasticModels.CNFModels import CNF
from lightgbm import LGBMRegressor


class MAFModel(ABC):
    """
    Abstract class for MAF models.

    Params:
    -------
    :param model_name: str - Name of the model
    :param config_path: str - Path to the configuration file        
    """
    def __init__(self, model_name, config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.model_name = model_name
        self.model_params = self.config[model_name]["parameters"]
        self.model_path = self.config[model_name]["model_path"]
        self.c_scaler_path = self.config[model_name]["c_scaler_path"]
        self.x_scaler_path = self.config[model_name]["x_scaler_path"]
        self.model = self.load_model(self.model_path)
        self.c_scaler = self.load_scaler(self.c_scaler_path)
        self.X_scaler = self.load_scaler(self.x_scaler_path)

    @abstractmethod
    def load_model(self, model_path):
        """
        Load model from the given path with the specified parameters.
        This method has to be implemented by each subclass.
        """
        pass

    def load_scaler(self, scaler_path):
        """
        Load scaler from the specified path.
        """
        return joblib.load(scaler_path)
    
    def predict(self, data: np.ndarray=None, steps: int=1, num_sample_paths: int=10):
        """
        Single-step prediction method reusing its previous predictions as input for the next step until the specified number of steps is reached.

        Params:
        -------
        :param data: np.ndarray - Input data for the first step
        :param steps: int - Number of steps to forecast
        :param num_sample_paths: int - Number of sample paths to generate
        """
        c_scaler = self.c_scaler
        X_scaler = self.X_scaler

        if not isinstance(data, np.ndarray):
            raise np.array(data)

        context = c_scaler.transform(data)
        c_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(1)     
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for i in range(steps):
                samples = self.model.generate_sample_paths(num_sample_paths=num_sample_paths, num_samples=i+1, context=c_tensor)
                samples_tensor = torch.tensor(samples, dtype=torch.float32)
                samples_tensor_mean = samples_tensor.mean(dim=0, keepdim=True)
                predictions.append(samples_tensor_mean[:, i, :].numpy().item())
                last_row = c_tensor[-1, :, :-1]
                new_row = torch.cat([samples_tensor_mean[:, i, :], last_row], dim=1)
                c_tensor = torch.cat((c_tensor, new_row.unsqueeze(0)), dim=0)

        predictions = np.array(predictions).reshape(-1, 1)
        rescaled_samples = X_scaler.inverse_transform(predictions)
        return rescaled_samples.flatten()

class LSTM_MAF(MAFModel):
    
    def __init__(self, model_name="lstm-maf", config_path="/Users/florian/Documents/github/thesis/stochastic-optimization/EnergyStorage_II/Forecast/forecast_models_config.yaml"):
        super().__init__(model_name, config_path)

    def load_model(self, model_path):
        lstm_params = self.model_params["lstm_parameters"]
        maf_params = self.model_params["maf_parameters"]
        try:
            context_model = Context_LSTM(
                input_size=lstm_params["input_size"],
                hidden_size=lstm_params["hidden_size"],
                num_layers=lstm_params["num_layers"],
                output_size=lstm_params["output_size"],
                dropout=lstm_params["dropout"]
            )
            model = Conditional_MAF(
                x_dim=maf_params["x_dim"],
                c_dim=maf_params["c_dim"],
                context_model=context_model,
                trainable_q0=maf_params["trainable_q0"],
                num_flows=maf_params["num_flows"],
                hidden_features=maf_params["hidden_features"],
                num_blocks=maf_params["num_blocks"],
                dropout_probability=maf_params["dropout_probability"],
                weight_decay=maf_params["weight_decay"],
                temperature=maf_params["temperature"],
                use_batch_norm=maf_params["use_batch_norm"],
                use_residual_blocks=maf_params["use_residual_blocks"],
                p=maf_params["p"]
            )
            model.load(model_path)
            model.eval()

        except Exception as e: 
            print(f"Failed to load model from {model_path}: {e}")
            raise 
            
        return model
    
class LSTM_MAF_old(MAFModel):
    
    def __init__(self, model_name="lstm-maf_old", config_path="/Users/florian/Documents/github/thesis/stochastic-optimization/EnergyStorage_II/Forecast/lstmmaf_config.yaml"):
        super().__init__(model_name, config_path)

    def load_model(self, model_path):
        lstm_params = self.model_params["lstm_parameters"]
        maf_params = self.model_params["maf_parameters"]
        try:
            context_model = Context_LSTM_old(
                input_size=lstm_params["input_size"],
                hidden_size=lstm_params["hidden_size"],
                num_layers=lstm_params["num_layers"],
                output_size=lstm_params["output_size"],
                dropout=lstm_params["dropout"]
            )
            model = CNFWithARModel(
                x_dim=maf_params["x_dim"],
                c_dim=maf_params["c_dim"],
                ar_model=context_model,
                num_flows=maf_params["num_flows"],
                hidden_features=maf_params["hidden_features"],
                num_blocks=maf_params["num_blocks"],
                dropout_probability=maf_params["dropout_probability"],
                weight_decay=maf_params["weight_decay"],
                temperature=maf_params["temperature"],
                use_batch_norm=maf_params["use_batch_norm"],
                use_residual_blocks=maf_params["use_residual_blocks"],
            )
            model.load(model_path)
            model.eval()

        except Exception as e: 
            print(f"Failed to load model from {model_path}: {e}")
            raise 
            
        return model

class RNN_MAF(MAFModel):
    
    def __init__(self, model_name="rnn-maf", config_path="/Users/florian/Documents/github/thesis/stochastic-optimization/EnergyStorage_II/Forecast/forecast_models_config.yaml"):
        super().__init__(model_name, config_path)

    def load_model(self, model_path):
        rnn_params = self.model_params["rnn_parameters"]
        maf_params = self.model_params["maf_parameters"]
        try:
            context_model = Context_RNN(
                input_size=rnn_params["input_size"],
                hidden_size=rnn_params["hidden_size"],
                num_layers=rnn_params["num_layers"],
                output_size=rnn_params["output_size"],
                dropout=rnn_params["dropout"]
            )
            model = Conditional_MAF(
                x_dim=maf_params["x_dim"],
                c_dim=maf_params["c_dim"],
                context_model=context_model,
                trainable_q0=maf_params["trainable_q0"],
                num_flows=maf_params["num_flows"],
                hidden_features=maf_params["hidden_features"],
                num_blocks=maf_params["num_blocks"],
                dropout_probability=maf_params["dropout_probability"],
                weight_decay=maf_params["weight_decay"],
                temperature=maf_params["temperature"],
                use_batch_norm=maf_params["use_batch_norm"],
                use_residual_blocks=maf_params["use_residual_blocks"],
                p=maf_params["p"]
            )
            model.load(model_path)
            model.eval()

        except Exception as e: 
            print(f"Failed to load model from {model_path}: {e}")
            raise 
            
        return model

class Transformer_MAF(MAFModel):
    
    def __init__(self, model_name="transformer-maf", config_path="/Users/florian/Documents/github/thesis/stochastic-optimization/EnergyStorage_II/Forecast/forecast_models_config.yaml"):
        super().__init__(model_name, config_path)

    def load_model(self, model_path):
        transformer_params = self.model_params["transformer_parameters"]
        maf_params = self.model_params["maf_parameters"]
        try:
            context_model = Context_Transformer(
                d_model=transformer_params["d_model"],
                nhead=transformer_params["nhead"],
                num_encoder_layers=transformer_params["num_encoder_layers"],
                dim_feedforward=transformer_params["dim_feedforward"],
                output_size=transformer_params["output_size"],
                dropout=transformer_params["dropout"]
            )
            model = Conditional_MAF(
                x_dim=maf_params["x_dim"],
                c_dim=maf_params["c_dim"],
                context_model=context_model,
                trainable_q0=maf_params["trainable_q0"],
                num_flows=maf_params["num_flows"],
                hidden_features=maf_params["hidden_features"],
                num_blocks=maf_params["num_blocks"],
                dropout_probability=maf_params["dropout_probability"],
                weight_decay=maf_params["weight_decay"],
                temperature=maf_params["temperature"],
                use_batch_norm=maf_params["use_batch_norm"],
                use_residual_blocks=maf_params["use_residual_blocks"],
                p=maf_params["p"]
            )
            model.load(model_path)
            model.eval()

        except Exception as e: 
            print(f"Failed to load model from {model_path}: {e}")
            raise 
            
        return model

class MLP_MAF(MAFModel):
    
    def __init__(self, model_name="mlp-maf", config_path="/Users/florian/Documents/github/thesis/stochastic-optimization/EnergyStorage_II/Forecast/forecast_models_config.yaml"):
        super().__init__(model_name, config_path)

    def load_model(self, model_path):
        mlp_params = self.model_params["mlp_parameters"]
        maf_params = self.model_params["maf_parameters"]
        try:
            context_model = Context_MLP(
                input_size=mlp_params["input_size"],
                hidden_size=mlp_params["hidden_size"],
                num_layers=mlp_params["num_layers"],
                output_size=mlp_params["output_size"],
                dropout=mlp_params["dropout"]
            )
            model = Conditional_MAF(
                x_dim=maf_params["x_dim"],
                c_dim=maf_params["c_dim"],
                context_model=context_model,
                trainable_q0=maf_params["trainable_q0"],
                num_flows=maf_params["num_flows"],
                hidden_features=maf_params["hidden_features"],
                num_blocks=maf_params["num_blocks"],
                dropout_probability=maf_params["dropout_probability"],
                weight_decay=maf_params["weight_decay"],
                temperature=maf_params["temperature"],
                use_batch_norm=maf_params["use_batch_norm"],
                use_residual_blocks=maf_params["use_residual_blocks"],
                p=maf_params["p"]
            )
            model.load(model_path)
            model.eval()

        except Exception as e: 
            print(f"Failed to load model from {model_path}: {e}")
            raise 
            
        return model

class CNFModel(MAFModel):
    
    def __init__(self, model_name="cnf", config_path="/Users/florian/Documents/github/thesis/stochastic-optimization/EnergyStorage_II/Forecast/forecast_models_config.yaml"):
        super().__init__(model_name, config_path)

    def load_model(self, model_path):
        cnf_params = self.model_params
        try:
            model = CNF(
                x_dim=cnf_params["x_dim"],
                c_dim=cnf_params["c_dim"],
                num_flows=cnf_params["num_flows"],
                hidden_features=cnf_params["hidden_features"],
                trainable_q0=cnf_params["trainable_q0"],
                use_batch_norm=cnf_params["use_batch_norm"],
                use_residual_blocks=cnf_params["use_residual_blocks"],
                num_blocks=cnf_params["num_blocks"],
                dropout_probability=cnf_params["dropout_probability"],
                temperature=cnf_params["temperature"],
                weight_decay=cnf_params["weight_decay"],
                lr=cnf_params["lr"]
            )
            model.load(model_path)
            model.eval()

        except Exception as e: 
            print(f"Failed to load model from {model_path}: {e}")
            raise 
            
        return model
    
    def predict(self, data, steps, num_sample_paths=10):
        c_scaler = self.c_scaler
        X_scaler = self.X_scaler

        if not isinstance(data, np.ndarray):
            raise np.array(data)

        context = c_scaler.transform(data)
        c_tensor = torch.tensor(context, dtype=torch.float32)     
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for i in range(steps):
                samples = self.model.generate_sample_paths(num_sample_paths=num_sample_paths, num_samples=i+1, context=c_tensor)
                samples_tensor = torch.tensor(samples, dtype=torch.float32)
                samples_tensor_mean = samples_tensor.mean(dim=0, keepdim=True)
                predictions.append(samples_tensor_mean[0, i, :].numpy().item())
                last_row = c_tensor[-1, :-1]
                new_row = torch.cat([samples_tensor_mean[0, i, :], last_row], dim=0)
                c_tensor = torch.cat((c_tensor, new_row.unsqueeze(0)), dim=0)

        predictions = np.array(predictions).reshape(-1, 1)
        rescaled_samples = X_scaler.inverse_transform(predictions)
        return rescaled_samples.flatten()

class LightGBMModel():
    
    def __init__(self, model_name="lightgbm", config_path="/Users/florian/Documents/github/thesis/stochastic-optimization/EnergyStorage_II/Forecast/forecast_models_config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.model_name = model_name
        self.model_config = self.config[model_name]
        self.model = self.load_model(self.model_config["model_path"])

    def load_model(self, model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            raise
        return model

    def predict(self, data, steps):
        current_data = data.copy()
        predictions = []  

        for _ in range(steps):
            forecast = self.model.predict(current_data)
            predictions.append(forecast)
            new_row = np.array([forecast])
            current_data = np.concatenate((current_data, new_row), axis=1)
            current_data = current_data[:, -24:]

        return np.array(predictions).flatten()





