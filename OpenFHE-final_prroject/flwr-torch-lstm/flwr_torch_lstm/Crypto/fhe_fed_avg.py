import time
from flwr_torch_lstm.Crypto.fhe_crypto import FheCryptoAPI

from typing import Callable, Dict, List, Optional, Tuple, Union
from logging import INFO, WARNING

import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    MetricsAggregationFn,
    ndarrays_to_parameters,
    parameters_to_ndarrays,)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr_torch_lstm.task import Net
import pickle


## Hyper-parameters 
input_size = 16 # dataset collumns
hidden_size = 1
num_layers = 3
num_classes = 2 # num y class

class FheFedAvg(FedAvg):
    def __init__(
        self,
        *,
        dataset_name: str = None,
        **kwargs,
    ) -> None:
        log(INFO, "FHE strategy created")

        self.model = Net(input_size, hidden_size, num_layers, num_classes)
        self.init_stage = True
        self.dataset_name = dataset_name
        self.ckpt_name = ""

        # Get FHE context and keys
        self.cc, self.pubkey, self.seckey = FheCryptoAPI.create_crypto_context_and_keys()

        super().__init__(**kwargs)

    def __decrypt_params(self, parameters: Parameters) -> NDArrays:
        log(INFO, "FHE decrypt params")
        return [
            FheCryptoAPI.decrypt_torch_tensor(
                self.cc, self.seckey,
                pickle.loads(p), v.dtype, v.shape
            ).cpu().numpy()
            for p, v in zip(parameters.tensors, self.model.state_dict().values())
        ]

    def __encrypt_params(self, ndarrays: NDArrays) -> Parameters:
        log(INFO, "FHE encrypt params")
        return Parameters(
            tensors=[
                pickle.dumps(FheCryptoAPI.encrypt_numpy_array(self.cc, self.pubkey, arr))
                for arr in ndarrays
            ],
            tensor_type=""
        )

    def __add_crypto_config(self, config: dict, server_round: int, skip: bool = False) -> None:
        config.update({
            'crypto_context': self.cc,
            'public_key': self.pubkey,
            'secret_key': self.seckey,
            'curr_round': server_round,
            'ds': self.dataset_name,
            'skip': skip,
        })

    def _save_checkpoint(self, params: Parameters):
        self.ckpt_name = f"ckpt_fhe_{int(time.time())}.bin"
        with open(self.ckpt_name, 'wb') as f:
            pickle.dump(params, f)

    def _load_previous_checkpoint(self) -> Parameters:
        with open(self.ckpt_name, 'rb') as f:
            return pickle.load(f)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        log(INFO, "FHE configure fit")

        if self.init_stage:
            parameters = self.__encrypt_params(parameters_to_ndarrays(parameters))
            self._save_checkpoint(parameters)
            self.init_stage = False
        elif len(parameters.tensors) == 0:
            parameters = self._load_previous_checkpoint()

        fit_config = super().configure_fit(server_round, parameters, client_manager)

        for _, ins in fit_config:
            self.__add_crypto_config(ins.config, server_round, skip=(len(parameters.tensors) == 0))

        return fit_config

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        log(INFO, "FHE configure eval")

        if self.init_stage:
            parameters = self.__encrypt_params(parameters_to_ndarrays(parameters))
            self.init_stage = False

        eval_config = super().configure_evaluate(server_round, parameters, client_manager)

        for _, ins in eval_config:
            self.__add_crypto_config(ins.config, server_round, skip=(len(parameters.tensors) == 0))

        return eval_config

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        log(INFO, "FHE evaluate")

        if self.evaluate_fn is None:
            return None

        ndarrays = parameters.tensors if self.init_stage else self.__decrypt_params(parameters)
        return self.evaluate_fn(server_round, ndarrays, {}) or None

    def __secure_aggregate(self, weights_results: List[Tuple[Parameters, int]]) -> Parameters:
        log(INFO, "FHE secure aggregate")

        num_total = sum(cnt for _, cnt in weights_results)
        fractions = [cnt / num_total for _, cnt in weights_results]
        updates = [params.tensors for params, _ in weights_results]

        aggregated = FheCryptoAPI.secure_fedavg_aggregate_updates(self.cc, updates, fractions)
        return Parameters(tensors=aggregated, tensor_type="")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        log(INFO, "FHE aggregate fit")

        stragglers_mask = [res.metrics.get("is_straggler", False) for _, res in results]

        if sum(stragglers_mask) > 0:
            log(WARNING, f"Found {sum(stragglers_mask)} stragglers; discarding their updates.")

        weights_results = [
            (res.parameters, res.num_examples)
            for i, (_, res) in enumerate(results)
            if not stragglers_mask[i]
        ]

        if not weights_results:
            return None, {}

        parameters_aggregated = self.__secure_aggregate(weights_results)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn and parameters_aggregated.tensors:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1: # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        if parameters_aggregated.tensors:
            self._save_checkpoint(parameters_aggregated)

        return parameters_aggregated, metrics_aggregated
