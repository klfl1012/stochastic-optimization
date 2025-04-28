import numpy as np
from typing import Literal, Union
import sys
sys.path.append("../")
from BaseClasses.SDPModel import SDPModel
from Forecast.ForecastModels import MLP_MAF, RNN_MAF, LSTM_MAF_old, LSTM_MAF, LightGBMModel, CNFModel

class EnergyStorageModel(SDPModel):
    """
    Energy Storage Model with forecasted day-ahead prices. Prices constists of day-ahead prices and the model is discretized on an daily basis.

    See inherited class for docstring of other parameters 

    :param init_args: dict - contains the information to populate initial state, including eta (the fraction of
    :param exog_params: dict - all the parameters including DataFrame (exog_data) containning the price information
    :param possible_decisions: list - list of possible decisions we could make
    """
    def __init__(
        self,
        state_names: list=["energy_amount", "price"],
        decision_names: list=["buy", "hold", "sell"],
        S0: dict=None,
        t0: int=0,
        T: int=None,
        seed: int=0,
        init_args: dict={"eta": None, "Rmax": None, "max_load_per_hour": None},
        exog_params: dict={"hist_price": None},
        possible_decisions: list=[{"buy": 1, "hold": 0, "sell": 0}, {"buy": 0, "hold": 1, "sell": 0}, {"buy": 0, "hold": 0, "sell": 1}],
        model_name: Literal["lstm-maf_old", "lstm-maf", "lstm-maf-24", "rnn-maf", "mlp-maf", "lightgbm", "cnf", "cnf-24", "norm"]= "cnf-24",
    ) -> None:
        assert isinstance(init_args.get("eta"), float) and 0 < init_args.get("eta") <= 1, "eta must be float and 0 < eta <= 1"
        assert isinstance(init_args.get("Rmax"), Union[int, float]), "Rmax must be a int / float"
        assert isinstance(init_args.get("max_load_per_hour"), Union[int, float]), "max_load_per_hour must be a int / float"

        self.init_args = init_args
        self.exog_params = exog_params
        self.possible_decisions = possible_decisions
        self.T = T
        self.S0 = S0

        model_mapping = {
            "lstm-maf_old": LSTM_MAF_old,
            "lstm-maf": LSTM_MAF,
            "lstm-maf-24": LSTM_MAF,
            "rnn-maf": RNN_MAF,
            "mlp-maf": MLP_MAF,
            "lightgbm": LightGBMModel,
            "cnf": CNFModel,
            "cnf-24": CNFModel
        }
        if model_name == "norm":
            self.price_model = None
        else:
            self.price_model = model_mapping[model_name](model_name=model_name)
        
        if S0 is None:
            S0 = self.sample_initial_state()

        super().__init__(state_names, decision_names, S0, t0, T, seed)

    def sample_initial_state(self):
        """
        Create random historic and forecast price values  
        """
        S0 = {
            "energy_amount": self.init_args["Rmax"] / 2,
            "price": self.exog_params["hist_price"][0],
        }
        return S0

    def get_price_forecast(self, forecast_length):
        """
        Returns the forecasted prices for the given forecat_length
        """
        if self.price_model == None:
            mean = self.state.price.mean()
            prediction = np.random.normal(loc=mean, scale=2, size=forecast_length)

        elif self.price_model.model_name in ["lstm-maf_old", "lstm-maf-24", "cnf-24", "lightgbm"]:
            last_prices = np.array(self.state.price).reshape(1, -1)
            prediction = self.price_model.predict(data=last_prices, steps=forecast_length)

        else:
            last_prices = np.array(self.state.price[-2:]).reshape(1, -1)
            prediction = self.price_model.predict(data=last_prices, steps=forecast_length)
        return prediction
    
    def build_decision(self, decisions):
        """
        Builds decisions for a vector of prices and energy constraints.
        :param decisions: list of dicts - contains "buy", "sell", "hold" for each time step
        :return: list of namedtuple decisions
        """
        max_load_per_hour = self.init_args["max_load_per_hour"] * self.init_args["eta"]
        max_energy = self.init_args["Rmax"]
        energy_amount = self.state.energy_amount

        decision_vector = []
        for decision in decisions:
            info_copy = {"buy": 0, "hold": 0, "sell": 0}
            for k in self.decision_names:
                if k == "buy" and decision[k] > 0:
                    max_buy_amount = max_energy - energy_amount
                    buy_amount = max(0, min(max_buy_amount, max_load_per_hour))
                    info_copy[k] = buy_amount
                    energy_amount += buy_amount
                elif k == "sell" and decision[k] > 0:
                    sell_amount = min(energy_amount, max_load_per_hour)
                    info_copy[k] = sell_amount
                    energy_amount -= sell_amount
                else:
                    info_copy[k] = 0
            decision_vector.append(self.Decision(*[info_copy[k] for k in self.decision_names]))
        return decision_vector
    
    def exog_info_fn(self, decision):
        """
        Give next price based on the current time step
        """
        next_price = self.exog_params["hist_price"][self.t]
        exog_info = {
            "price": next_price,
        }
        return exog_info

    def transition_fn(self, decision, exog_info):
        """
        Describe how the state (energy_amount and price) evolve based on the decision made and exogenous information
        :params decision: namedtuple - decision object
        :params exog_info: dict - contains exogenous information
        """
        sold_amount = sum([d.sell for d in decision])
        bought_amount = sum([d.buy for d in decision])

        new_energy_amount = (
            self.state.energy_amount + bought_amount - sold_amount
        )
        new_prices = exog_info["price"]

        new_state = {
            "energy_amount": new_energy_amount,
            "price": new_prices,
        #     "price_tp1": new_price_forecast
        }
        return new_state

    def objective_fn(self, decision, exog_info):
        """
        Calculate the contribution at time t
        """
        obj_part = 0
        for i, decision in enumerate(decision):
            price = self.state.price[i]
            sell_energy_mwh = decision.sell/ 1000
            buy_energy_mwh = (decision.buy / self.init_args["eta"]) / 1000
            obj_part += price * (sell_energy_mwh - buy_energy_mwh)
        
        return obj_part















# class EnergyStorageModel(SDPModel):

#     def __init__(self, state_names, decision_names, S0, t0, T, seed, init_args, exog_params, possible_decisions) -> None:
#         """
#         See inherited class for docstring of other parameters 

#         :param init_args: dict - contains the information to populate initial state, including eta (the fraction of
#         :param exog_params: dict - all the parameters including DataFrame (exog_data) containning the price information
#         :param possible_decisions: list - list of possible decisions we could make
#         """
#         super().__init__(state_names, decision_names, S0, t0, T, seed)
#         self.init_args = init_args
#         self.exog_params = exog_params
#         self.possible_decisions = possible_decisions

#     def build_decision(self, info):
#         """
#         this function returns a decision

#         :param info: dict - contains all decision info
#         :return: namedtuple - a decision object
#         """
#         energy_amount = self.state.energy_amount
#         max_load_per_hour = self.init_args["max_load_per_hour"] * self.init_args["eta"]
#         info_copy = {"buy": 0, "hold": 0, "sell": 0}
#         # the amount of power that can be bought or sold is limited by constraints
#         for k in self.decision_names:
#             if k == "buy" and info[k] > 0:
#                 max_buy_amount = (self.init_args["Rmax"] - energy_amount)
#                 info_copy[k] = max(0, min(max_buy_amount, max_load_per_hour)) 
#             elif k == "sell" and info[k] > 0:
#                 info_copy[k] = min(energy_amount, max_load_per_hour)
#             else:
#                 info_copy[k] = info[k]
#         return self.Decision(*[info_copy[k] for k in self.decision_names])

#     def exog_info_fn(self, decision):
#         """
#         Give next price based on the current time step.
#         """
#         next_price = self.exog_params["hist_price"][self.t]
        
#         return {"price": next_price}

#     def transition_fn(self, decision, exog_info):
#         """
#         Describe how the state (energy_amount and price) evolve based on the decision made and exogenous information
#         :params decision: namedtuple - decision object
#         :params exog_info: dict - contains exogenous information
#         """
#         new_energy_amount = (
#             self.state.energy_amount + decision.buy - decision.sell
#         )
#         return {"energy_amount": new_energy_amount, "price": exog_info["price"]}

#     def objective_fn(self, decision, exog_info):
#         """
#         Calculate the contribution at time t
#         """
#         sell_energy_mwh = decision.sell / 1000
#         buy_energy_mwh = (decision.buy / self.init_args["eta"]) / 1000
#         obj_part = self.state.price * (sell_energy_mwh - buy_energy_mwh)
#         return obj_part



# class EnergyStorageModel_2(EnergyStorageModel):
#     """
#     Still scenario 1 but with forecasted prices 
#     """

#     def __init__(
#         self,
#         state_names: list=["energy_amount", "price", "price_tm1", "price_tm2", "price_tm3"],
#         decision_names: list=["buy", "hold", "sell"],
#         S0: dict=None,
#         t0: int=0,
#         T: int=None,
#         seed: int=0,
#         init_args: dict=None,
#         exog_params: dict=None,
#         possible_decisions: dict=None,
#     ) -> None:
#         """
#         See inherited class for docstring of other parameters 

#         :param init_args: dict - contains the information to populate initial state, including eta (the fraction of
#         :param exog_params: dict - all the parameters including DataFrame (exog_data) containning the price information
#         :param possible_decisions: list - list of possible decisions we could make
#         """
#         self.init_args = init_args
#         self.exog_params = exog_params
#         self.possible_decisions = possible_decisions
#         # self.price_model = DAP(model_name= self.init_args["price_model"])
#         self.forecast_length = self.init_args["forecast_length"]
#         # for i in range(self.forecast_length):
#         #     state_name = f"price_tp{i+1}"
#         #     if state_name not in state_names:
#         #         state_names.append(state_name)
#         if S0 is None:
#             S0 = self.sample_initial_state()

#         super().__init__(state_names, decision_names, S0, t0, T, seed, init_args, exog_params, possible_decisions)

#     def sample_initial_state(self):
#         """
#         Create random historic and forecast price values  
#         """
#         sampled_hist_prices = np.random.choice(self.exog_params["hist_price"], 3, replace=False)
#         price_forecast_init_state = self.price_model.make_forecast(data=sampled_hist_prices, steps=self.forecast_length)
#         S0 = {
#             "energy_amount": self.init_args["Rmax"] / 2,
#             "price": self.exog_params["hist_price"][0],
#             "price_tm1": sampled_hist_prices[0],
#             "price_tm2": sampled_hist_prices[1],
#             "price_tm3": sampled_hist_prices[2],
#         }
#         for i in range(self.forecast_length):
#             S0[f"price_tp{i+1}"] = price_forecast_init_state[i]
#         return S0
    
#     def get_last_prices(self):
#         """
#         Returns a np.array of the t-forecast_length prices
#         """
#         # forecast_length = self.forecast_length für spätere Implementierung von mehr als 3 lag prices im state -> Hyperparam 
#         # replace 3 mit forecast_length
#         if self.t == 0:
#             return np.array([self.exog_params["hist_price"][0]] * 3)
#         elif self.t < 3:
#             return np.array(self.exog_params["hist_price"][:self.t].tolist() + [self.exog_params["hist_price"][0]] * (3 - self.t))
#         return np.array(self.exog_params["hist_price"][self.t-3:self.t])

#     def get_price_forecast(self, forecast_length):
#         """
#         Returns the forecasted prices for the given forecat_length
#         """
#         # last_prices = self.get_last_prices() # dont use last prices function use state values
#         last_prices = np.array([self.state.price_tm3, self.state.price_tm2, self.state.price_tm1])
#         return self.price_model.make_forecast(data=last_prices, steps=forecast_length)
    
#     def exog_info_fn(self, decision):
#         """
#         Give next price based on the current time step
#         """
#         next_price = self.exog_params["hist_price"][self.t]
#         price_forecast = self.get_price_forecast(forecast_length=self.forecast_length)
#         exog_info = {
#             "price": next_price
#         }
#         for i in range(self.forecast_length):
#             exog_info[f"price_tp{i+1}"] = price_forecast[i]
#         return exog_info

#     def transition_fn(self, decision, exog_info):
#         """
#         Describe how the state (energy_amount and price) evolve based on the decision made and exogenous information
#         :params decision: namedtuple - decision object
#         :params exog_info: dict - contains exogenous information
#         """
#         new_energy_amount = (
#             self.state.energy_amount + decision.buy - decision.sell
#         )
#         # tbd: 
#         # hist_prices = self.get_last_prices()
#         # new_price_tm1, new_price_tm2, new_price_tm3 = hist_prices[0], hist_prices[1], hist_prices[2]

#         new_state = {
#             "energy_amount": new_energy_amount,
#             "price_tm3": self.state.price_tm2,
#             "price_tm2": self.state.price_tm1,
#             "price_tm1": self.state.price,
#             "price": exog_info["price"],
#         }
#         for i in range(self.forecast_length):
#             new_state[f"price_tp{i+1}"] = exog_info[f"price_tp{i+1}"]
#         return new_state

#     def objective_fn(self, decision, exog_info):
#         """
#         Calculate the contribution at time t
#         """
#         sell_energy_mwh = decision.sell / 1000
#         buy_energy_mwh = (decision.buy / self.init_args["eta"]) / 1000
#         obj_part = self.state.price * (sell_energy_mwh - buy_energy_mwh)
#         return obj_part
    




# class EnergyStorageModel(SDPModel):
#     """
#     Energy Storage Model with forecasted day-ahead prices. Prices constists of day-ahead prices and the model is discretized on an daily basis.

#     See inherited class for docstring of other parameters 

#     :param init_args: dict - contains the information to populate initial state, including eta (the fraction of
#     :param exog_params: dict - all the parameters including DataFrame (exog_data) containning the price information
#     :param possible_decisions: list - list of possible decisions we could make
#     """
#     def __init__(
#         self,
#         state_names: list=["energy_amount", "price"],
#         decision_names: list=["buy", "hold", "sell"],
#         S0: dict=None,
#         t0: int=0,
#         T: int=None,
#         seed: int=0,
#         init_args: dict={"eta": None, "Rmax": None, "max_load_per_hour": None},
#         exog_params: dict={"hist_price": None},
#         possible_decisions: list=[{"buy": 1, "hold": 0, "sell": 0}, {"buy": 0, "hold": 1, "sell": 0}, {"buy": 0, "hold": 0, "sell": 1}],
#         model_name: Literal["lstm-maf_old", "lstm-maf", "lstm-maf-24", "rnn-maf", "mlp-maf", "lightgbm", "cnf", "cnf-24"]= "lstm-maf",
#     ) -> None:
        
#         self.init_args = init_args
#         self.exog_params = exog_params
#         self.possible_decisions = possible_decisions
#         self.T = T
#         self.S0 = S0

#         model_mapping = {
#             "lstm-maf_old": LSTM_MAF_old,
#             "lstm-maf": LSTM_MAF,
#             "lstm-maf-24": LSTM_MAF,
#             "rnn-maf": RNN_MAF,
#             "mlp-maf": MLP_MAF,
#             "lightgbm": LightGBMModel,
#             "cnf": CNFModel,
#             "cnf-24": CNFModel,
#         }
#         self.price_model = model_mapping[model_name](model_name=model_name)
        
#         if S0 is None:
#             S0 = self.sample_initial_state()

#         super().__init__(state_names, decision_names, S0, t0, T, seed)

#     def sample_initial_state(self):
#         """
#         Create random historic and forecast price values  
#         """
#         S0 = {
#             "energy_amount": self.init_args["Rmax"] / 2,
#             "price": self.exog_params["hist_price"][0],
#         }
#         return S0

#     def get_price_forecast(self, forecast_length):
#         """
#         Returns the forecasted prices for the given forecat_length
#         """
#         if self.price_model.model_name in ["lstm-maf_old", "lstm-maf-24", "cnf-24"]:
#             last_prices = np.array(self.state.price).reshape(1, -1)
#             prediction = self.price_model.predict(data=last_prices, steps=forecast_length)
#         else:
#             last_prices = np.array(self.state.price[-2:]).reshape(1, -1)
#             prediction = self.price_model.predict(data=last_prices, steps=forecast_length)
#         return prediction
    
#     def build_decision(self, decisions):
#         """
#         Builds decisions for a vector of prices and energy constraints.
#         :param decisions: list of dicts - contains "buy", "sell", "hold" for each time step
#         :return: list of namedtuple decisions
#         """
#         max_load_per_hour = self.init_args["max_load_per_hour"] * self.init_args["eta"]
#         max_energy = self.init_args["Rmax"]
#         energy_amount = self.state.energy_amount

#         decision_vector = []
#         for decision in decisions:
#             info_copy = {"buy": 0, "hold": 0, "sell": 0}
#             for k in self.decision_names:
#                 if k == "buy" and decision[k] > 0:
#                     max_buy_amount = max_energy - energy_amount
#                     buy_amount = max(0, min(max_buy_amount, max_load_per_hour))
#                     info_copy[k] = buy_amount
#                     energy_amount += buy_amount
#                 elif k == "sell" and decision[k] > 0:
#                     sell_amount = min(energy_amount, max_load_per_hour)
#                     info_copy[k] = sell_amount
#                     energy_amount -= sell_amount
#                 else:
#                     info_copy[k] = 0
#             decision_vector.append(self.Decision(*[info_copy[k] for k in self.decision_names]))
#         return decision_vector
    
#     def exog_info_fn(self, decision):
#         """
#         Give next price based on the current time step
#         """
#         next_price = self.exog_params["hist_price"][self.t]
#         exog_info = {
#             "price": next_price,
#         }
#         return exog_info

#     def transition_fn(self, decision, exog_info):
#         """
#         Describe how the state (energy_amount and price) evolve based on the decision made and exogenous information
#         :params decision: namedtuple - decision object
#         :params exog_info: dict - contains exogenous information
#         """
#         sold_amount = sum([d.sell for d in decision])
#         bought_amount = sum([d.buy for d in decision])

#         new_energy_amount = (
#             self.state.energy_amount + bought_amount - sold_amount
#         )
#         new_prices = exog_info["price"]

#         new_state = {
#             "energy_amount": new_energy_amount,
#             "price": new_prices,
#         #     "price_tp1": new_price_forecast
#         }
#         return new_state

#     def objective_fn(self, decision, exog_info):
#         """
#         Calculate the contribution at time t
#         """
#         obj_part = 0
#         for i, decision in enumerate(decision):
#             price = self.state.price[i]
#             sell_energy_mwh = decision.sell/ 1000
#             buy_energy_mwh = (decision.buy / self.init_args["eta"]) / 1000
#             obj_part += price * (sell_energy_mwh - buy_energy_mwh)
        
#         return obj_part

