import time
from flwr_torch_lstm.Crypto.fhe_crypto import FheCryptoAPI

from typing import Dict, List, Optional, Tuple, Union
from logging import INFO, WARNING
import numpy as np

from flwr.common.logger import log
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
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
        self.ckpt_name = ""

        # Get FHE context and keys
        self.cc, self.pubkey, self.seckey = FheCryptoAPI.create_crypto_context_and_keys()

        super().__init__(**kwargs)

    def __decrypt_params(self, parameters: Parameters) -> NDArrays:
        log(INFO, "FHE decrypt params")
        decrypted_params = []
        
        for i, (param_data, model_param) in enumerate(zip(parameters.tensors, self.model.state_dict().values())):
            if isinstance(param_data, bytes):
                # If it's bytes, unpickle first
                encrypted_blocks = pickle.loads(param_data)
            else:
                # If it's already a list of encrypted blocks
                encrypted_blocks = param_data
                print("Parameters NOT encrypted")
                
            decrypted_tensor = FheCryptoAPI.decrypt_torch_tensor(
                self.cc, self.seckey,
                encrypted_blocks, model_param.dtype, model_param.shape
            )
            decrypted_params.append(decrypted_tensor.cpu().numpy())
        
        return decrypted_params

    def __encrypt_params(self, ndarrays: NDArrays) -> Parameters:
        log(INFO, "FHE encrypt params")
        encrypted_tensors = []
        
        for arr in ndarrays:
            encrypted_blocks = FheCryptoAPI.encrypt_numpy_array(self.cc, self.pubkey, arr)
            # Store as pickled bytes for consistency
            encrypted_tensors.append(pickle.dumps(encrypted_blocks))
            
        return Parameters(
            tensors=encrypted_tensors,
            tensor_type=""
        )

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
            # Convert initial parameters to encrypted format
            parameters = self.__encrypt_params(parameters_to_ndarrays(parameters))
            # self._save_checkpoint(parameters)
            self.init_stage = False
        elif len(parameters.tensors) == 0:
            # Load from checkpoint if no parameters
            parameters = self._load_previous_checkpoint()

        fit_config = super().configure_fit(server_round, parameters, client_manager)

        # Deliver keys, info to clients
        for client, fit_ins in fit_config:
            fit_ins.config['crypto_context'] = self.cc
            fit_ins.config['public_key'] = self.pubkey
            fit_ins.config['secret_key'] = self.seckey
            fit_ins.config['curr_round'] = server_round
            fit_ins.config['skip'] = (len(parameters.tensors) == 0)
        log(INFO, "FHE configure fit DONE")
        return fit_config

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        log(INFO, "FHE configure eval")

        if self.init_stage:
            parameters = self.__encrypt_params(parameters_to_ndarrays(parameters))
            self.init_stage = False
        eval_config = super().configure_evaluate(server_round, parameters, client_manager)

        for _, fit_ins in eval_config:
            fit_ins.config['crypto_context'] = self.cc
            fit_ins.config['public_key'] = self.pubkey
            fit_ins.config['secret_key'] = self.seckey
            fit_ins.config['skip'] = (len(parameters.tensors) == 0)
        log(INFO, "FHE eval fit DONE")
        return eval_config

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        log(INFO, "FHE evaluate")

        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        # Decrypt parameters for evaluation if they are encrypted
        if self.init_stage:
            parameters_ndarrays = parameters_to_ndarrays(parameters)
        else:
            parameters_ndarrays = self.__decrypt_params(parameters)
            
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        log(INFO, "FHE aggregate fit")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)  # parameters stay encrypted
            for _, fit_res in results
        ]
        
        # Convert results
        aggregated_ndarrays = FheCryptoAPI.aggregate_encrypted(weights_results, self.cc)
        parameters_aggregated = Parameters(tensors=aggregated_ndarrays, tensor_type="numpy.ndarray")

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn and len(parameters_aggregated.tensors) > 0:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1: # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # # Save checkpoint
        # if len(parameters_aggregated.tensors) > 0:
        #     self._save_checkpoint(parameters_aggregated)
        return parameters_aggregated, metrics_aggregated