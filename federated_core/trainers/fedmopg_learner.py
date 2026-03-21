# federated_core/trainers/fedmopg_learner.py

from typing import Dict, List, Any
import torch

from ..base_federated_learner import BaseFederatedLearner
from ..fed_utils import average_weights


class FedMoPGLearner(BaseFederatedLearner):
    """FedMoPG implementation with standard full parameter aggregation."""

    def _get_trainer_specific_attributes(self) -> Dict[str, Any]:
        return {}

    def _initialize_components_storage(self) -> None:
        self.client_weights["components"] = {
            "prompt_learner": [{} for _ in range(self.cfg.DATASET.USERS)]
        }

    def aggregate_models(self, client_ids: List[int]) -> Dict[str, torch.Tensor]:
        global_params = average_weights(
            self.client_weights["components"]["prompt_learner"],
            client_ids,
            self.client_data_sizes,
            islist=False,
        )
        return global_params

    def update_client_models(
        self,
        client_ids: List[int],
        global_component: Dict[str, torch.Tensor],
        all_clients: List[int] = None,
    ) -> None:
        if all_clients is None:
            all_clients = client_ids

        for client_id in all_clients:
            client_weight = self.client_weights["local"][client_id]
            for key, value in global_component.items():
                client_weight[key] = value
            print(f"Client {client_id} FedMoPG update completed")

    def _store_components(
        self, client_id: int, trained_state_dict: Dict[str, torch.Tensor]
    ) -> None:
        trainable_params = self._get_trainable_params(state_dict=trained_state_dict)
        self.client_weights["components"]["prompt_learner"][client_id] = trainable_params
