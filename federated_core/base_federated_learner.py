# federated_core/base_federated_learner.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
import copy
import time
import os

from Dassl.dassl.engine import build_trainer
from .fed_utils import (
    count_parameters, generate_performance_table, setup_federated_environment,
    initialize_logs, print_domain_distribution, validate_client_weights
)

class BaseFederatedLearner(ABC):
    """Base class for federated learning, defining common workflow and interfaces."""

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

        setup_federated_environment(cfg, args)

        self._setup_dataset_attributes()

        self.trainer = self._initialize_trainer()
        self.model_save_path = args.output_dir
        self.start_time = time.time()
        self.current_round = 0
        self.num_comm_rounds = cfg.OPTIM.ROUND

        self.client_data_sizes = []

        # Weight storage structure
        self.client_weights = {
            'global': {},
            'local': [{} for _ in range(cfg.DATASET.USERS)],
            'components': {},  # Initialized by subclasses
            'best': [{} for _ in range(cfg.DATASET.USERS)]
        }

        # Performance metrics storage
        self.metrics = {
            'local': {
                'acc': [0.0] * cfg.OPTIM.ROUND,
                'error': [0.0] * cfg.OPTIM.ROUND,
                'f1': [0.0] * cfg.OPTIM.ROUND,
                'client_acc': [[0.0] * cfg.OPTIM.ROUND for _ in range(cfg.DATASET.USERS)]
            },
            'base': {
                'acc': [0.0] * cfg.OPTIM.ROUND,
                'error': [0.0] * cfg.OPTIM.ROUND,
                'f1': [0.0] * cfg.OPTIM.ROUND,
                'client_acc': [[0.0] * cfg.OPTIM.ROUND for _ in range(cfg.DATASET.USERS)]
            },
            'new': {
                'acc': [0.0] * cfg.OPTIM.ROUND,
                'error': [0.0] * cfg.OPTIM.ROUND,
                'f1': [0.0] * cfg.OPTIM.ROUND,
                'client_acc': [[0.0] * cfg.OPTIM.ROUND for _ in range(cfg.DATASET.USERS)]
            }
        }

        self.time = [0.0] * cfg.OPTIM.ROUND
        self.rounds = list(range(cfg.OPTIM.ROUND))

        self.best_acc = 0.0
        self.best_round = -1
        self.best_base_acc = 0.0
        self.best_base_round = -1

        self.debug_mode = getattr(cfg, 'DEBUG_MODE', False)
        if self.debug_mode:
            print("Debug mode enabled: only test on final round")

        trainer_attrs = self._get_trainer_specific_attributes()
        for attr_name, attr_value in trainer_attrs.items():
            setattr(self, attr_name, attr_value)

        self._initialize_logs()

    # === Abstract methods: must be implemented by subclasses ===

    @abstractmethod
    def aggregate_models(self, client_ids: List[int]) -> Any:
        """Aggregate client models."""
        pass

    @abstractmethod
    def update_client_models(self, client_ids: List[int],
                           global_component: Any,
                           all_clients: Optional[List[int]] = None) -> None:
        """Update client models with aggregated results."""
        pass

    @abstractmethod
    def _store_components(self, client_id: int, trained_state_dict: Dict[str, torch.Tensor]) -> None:
        """Extract and store specific components from trained model state."""
        pass

    @abstractmethod
    def _initialize_components_storage(self) -> None:
        """Initialize component storage structure."""
        pass

    @abstractmethod
    def _get_trainer_specific_attributes(self) -> Dict[str, Any]:
        """Get trainer-specific attributes for initialization."""
        pass

    def has_identical_client_params(self) -> bool:
        """Check if all clients have identical parameters. Default: False."""
        return False

    # === Template methods: define common workflow ===

    def train(self) -> None:
        """Main federated learning training workflow (template method)."""
        print(f"Starting federated learning training, {self.num_comm_rounds} communication rounds")

        self.initialize_model_weights()

        start_time = time.time()

        for round_idx in range(self.num_comm_rounds):
            self.current_round = round_idx + 1
            print(f"\n============= Communication Round {round_idx+1}/{self.num_comm_rounds} =============")

            selected_client_ids = self.select_clients(round_idx)

            self.train_clients(selected_client_ids, round_idx)

            global_component = self.aggregate_models(selected_client_ids)

            self._update_clients_with_strategy(selected_client_ids, global_component)

            self._evaluate_with_strategy(selected_client_ids, round_idx)

        self._finalize_training()

    def test(self) -> None:
        """Evaluate model without training (template method)."""
        print(f"Loading model from {self.args.model_dir}")

        saved_checkpoint = self._load_saved_model()

        self._distribute_loaded_weights(saved_checkpoint)

        self._execute_evaluation()

        self._generate_test_report()

    # === Common methods: shared by all subclasses ===

    def initialize_model_weights(self) -> None:
        """Initialize model weights."""
        self._record_client_data_sizes()

        trainable_params = self._get_trainable_params()
        self.client_weights['global'] = trainable_params

        for client_id in range(self.cfg.DATASET.USERS):
            self.client_weights['local'][client_id] = copy.deepcopy(trainable_params)

        self._initialize_components_storage()

        print(f"Initialized weights for all {self.cfg.DATASET.USERS} clients")

    def select_clients(self, round_idx: int = 0) -> List[int]:
        """Select clients for current training round."""
        round_seed = self.cfg.SEED + round_idx if self.cfg.SEED >= 0 else None
        if round_seed is not None:
            orig_state = np.random.get_state()
            np.random.seed(round_seed)

        m = max(int(self.cfg.DATASET.FRAC * self.cfg.DATASET.USERS), 1)

        idxs_users = np.random.choice(range(self.cfg.DATASET.USERS), m, replace=False)

        if round_seed is not None:
            np.random.set_state(orig_state)

        print(f"Selected {len(idxs_users)}/{self.cfg.DATASET.USERS} clients: {idxs_users}")

        print_domain_distribution(idxs_users.tolist(), self.cfg)

        return idxs_users.tolist()

    def train_clients(self, client_ids: List[int], round_idx: int) -> None:
        """Train selected clients."""
        for client_id in client_ids:
            self._load_client_weights_for_training(client_id, round_idx)

            self._execute_client_training(client_id, round_idx)

            trained_state_dict = self.trainer.model.state_dict()
            self._store_components(client_id, trained_state_dict)

        print(f"------------Local training completed, Round: {round_idx}-------------")

    def evaluate_clients(self, client_ids: List[int], round_idx: int,
                        is_global: bool = False) -> float:
        """Evaluate client model performance."""
        data_type, metrics_key = self._determine_evaluation_mode(is_global)

        print(f"------------{data_type} dataset testing started-------------")

        results = []
        evaluated_client_ids = []

        # For global test with identical client params, test only once
        if is_global and not self.is_special_dataset and self.has_identical_client_params():
            print("Global test mode: using global test set")
            test_client_id = client_ids[0]
            self._load_trainable_params(self.trainer.model,
                                      self.client_weights['local'][test_client_id])

            client_result = self.trainer.test(global_test=True, current_epoch=round_idx)

            for client_id in client_ids:
                self.metrics[metrics_key]['client_acc'][client_id][round_idx] = client_result[0]
                results.append(client_result)
                evaluated_client_ids.append(client_id)
        else:
            for client_id in client_ids:
                self._load_trainable_params(self.trainer.model,
                                          self.client_weights['local'][client_id])

                if is_global and not self.is_special_dataset:
                    client_result = self.trainer.test(global_test=True, current_epoch=round_idx)
                else:
                    client_result = self.trainer.test(idx=client_id, current_epoch=round_idx)

                self.metrics[metrics_key]['client_acc'][client_id][round_idx] = client_result[0]
                results.append(client_result)
                evaluated_client_ids.append(client_id)

        avg_acc = self._process_evaluation_results(results, metrics_key, round_idx,
                                                 client_ids=evaluated_client_ids)

        if not is_global or self.is_special_dataset:
            self.time[round_idx] = time.time() - self.start_time

        print(f"------------{data_type} dataset testing completed-------------")
        return avg_acc

    def save_model(self, client_weights: Dict, filename: str) -> None:
        """Save model weights."""
        save_path = os.path.join(self.args.output_dir, filename)
        torch.save(client_weights, save_path)

        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path) / (1024 * 1024)
            print(f"Model saved to {save_path}, size: {file_size:.2f} MB")
        else:
            print(f"Model save failed: {save_path}")

    # === Protected methods: can be overridden by subclasses ===

    def _load_client_weights_for_training(self, client_id: int, round_idx: int) -> None:
        """Load client weights for training."""
        if round_idx == 0 or self.args.model in ["local", "fedavg", "fedprox"]:
            if self.args.model == "fedprox" and round_idx > 0:
                self._load_trainable_params(self.trainer.model, self.client_weights['global'])
            else:
                self._load_trainable_params(self.trainer.model, self.client_weights['global'])
        else:
            self._load_trainable_params(self.trainer.model,
                                      self.client_weights['local'][client_id])

    def _execute_client_training(self, client_id: int, round_idx: int) -> None:
        """Execute client training."""
        if self.args.model == "fedprox" and round_idx > 0:
            self.trainer.train(idx=client_id, global_epoch=round_idx, is_fed=True,
                             global_weight=self.client_weights['global'],
                             fedprox=True, mu=self.args.mu, debug_mode=self.debug_mode)
        else:
            self.trainer.train(idx=client_id, global_epoch=round_idx, is_fed=True,
                             debug_mode=self.debug_mode)

    def _update_clients_with_strategy(self, selected_client_ids: List[int],
                                    global_component: Any) -> None:
        """Update clients based on strategy."""
        if self.is_special_dataset:
            self.update_client_models(selected_client_ids, global_component)
        else:
            all_clients = list(range(0, self.cfg.DATASET.USERS))
            self.update_client_models(selected_client_ids, global_component, all_clients)

    def _evaluate_with_strategy(self, selected_client_ids: List[int], round_idx: int) -> None:
        """Execute evaluation based on strategy."""
        is_final_round = round_idx == self.num_comm_rounds - 1
        should_test = not self.debug_mode or is_final_round

        if should_test:
            avg_acc = self.evaluate_clients(selected_client_ids, round_idx, is_global=False)

            if avg_acc > self.best_acc:
                self.best_acc = avg_acc
                self.best_round = round_idx
                best_client_weights = copy.deepcopy(self.client_weights['local'])
                self.save_model(best_client_weights, "best_model.pt")
                print(f"Saved best model, Round: {round_idx + 1}, Accuracy: {avg_acc}")

            if not self.is_special_dataset:
                all_clients = list(range(0, self.cfg.DATASET.USERS))
                self.evaluate_clients(all_clients, round_idx, is_global=True)
        else:
            print(f"Debug mode: skipping test for round {round_idx + 1}")
            self.time[round_idx] = time.time() - self.start_time

    def _initialize_logs(self) -> None:
        """Initialize log files."""
        initialize_logs(self.model_save_path, self.args.model, self.args)

    # === Private methods: internal implementation ===

    def _setup_dataset_attributes(self) -> None:
        """Setup dataset-related attributes."""
        if hasattr(self.cfg.DATASET, 'NAME'):
            dataset_name = self.cfg.DATASET.NAME.lower()
            self.is_cifar_dataset = dataset_name in ['cifar10', 'cifar100']
            self.is_domainnet_dataset = dataset_name == 'domainnet'
            self.is_office_dataset = dataset_name == 'office'
            self.is_special_dataset = self.is_cifar_dataset or self.is_domainnet_dataset or self.is_office_dataset
        else:
            self.is_cifar_dataset = False
            self.is_domainnet_dataset = False
            self.is_office_dataset = False
            self.is_special_dataset = False

    def _initialize_trainer(self):
        """Initialize trainer."""
        trainer = build_trainer(self.cfg)
        trainer.fed_before_train()

        print("Model trainable parameters:")
        count_parameters(trainer.model, "prompt_learner")
        count_parameters(trainer.model, "image_encoder")

        return trainer

    def _record_client_data_sizes(self) -> None:
        """Record client data sizes."""
        self.client_data_sizes = []

        if self.args.trainer != 'CLIP':
            for client_id in range(self.cfg.DATASET.USERS):
                if (client_id in self.trainer.fed_train_loader_x_dict and
                    self.trainer.fed_train_loader_x_dict[client_id] is not None):
                    data_size = len(self.trainer.fed_train_loader_x_dict[client_id].dataset)
                    self.client_data_sizes.append(data_size)
                else:
                    self.client_data_sizes.append(0)

        print(f"Client data distribution: {len(self.client_data_sizes)} clients total")
        if self.is_domainnet_dataset or self.is_office_dataset:
            print(f"Dataset type: {self.cfg.DATASET.NAME}")

    def _get_trainable_params(self, model=None, state_dict=None):
        """Extract trainable parameters from model."""
        if model is None:
            model = self.trainer.model
        if state_dict is None:
            state_dict = model.state_dict()

        trainable_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = copy.deepcopy(state_dict[name])

        return trainable_params

    def _load_trainable_params(self, model, params):
        """Load trainable parameters into model."""
        model_state_dict = model.state_dict()

        for key, value in params.items():
            if key in model_state_dict:
                model_state_dict[key] = value

        model.load_state_dict(model_state_dict, strict=False)
        return model_state_dict

    def _determine_evaluation_mode(self, is_global: bool) -> Tuple[str, str]:
        """Determine evaluation mode."""
        if self.is_special_dataset:
            data_type = "local"
            metrics_key = "local"
        else:
            data_type = "local" if not is_global else self.cfg.DATASET.SUBSAMPLE_CLASSES
            metrics_key = data_type if is_global else "local"

        return data_type, metrics_key

    def _process_evaluation_results(self, results: List[Tuple], metrics_key: str,
                                  round_idx: int, client_ids: Optional[List[int]] = None) -> float:
        """Process evaluation results and compute average metrics."""
        if not results:
            print("Warning: No evaluation results to process")
            self.metrics[metrics_key]['acc'][round_idx] = 0.0
            self.metrics[metrics_key]['error'][round_idx] = 0.0
            self.metrics[metrics_key]['f1'][round_idx] = 0.0
            return 0.0

        test_acc = [result[0] for result in results]
        test_error = [result[1] for result in results]
        test_f1 = [result[2] for result in results]

        avg_acc = sum(test_acc) / len(test_acc) if test_acc else 0.0
        avg_error = sum(test_error) / len(test_error) if test_error else 0.0
        avg_f1 = sum(test_f1) / len(test_f1) if test_f1 else 0.0

        self.metrics[metrics_key]['acc'][round_idx] = avg_acc
        self.metrics[metrics_key]['error'][round_idx] = avg_error
        self.metrics[metrics_key]['f1'][round_idx] = avg_f1

        # Print per-domain results for DomainNet and Office datasets
        condition = False
        if self.args.model in ["CLIP", "local"]:
            condition = (round_idx == self.num_comm_rounds - 1)
        else:
            condition = (round_idx >= 2)

        if (self.is_domainnet_dataset or self.is_office_dataset) and condition and self.cfg.DATASET.SPLIT_CLIENT:
            domains = {"DomainNet":["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
                       "Office":["amazon", "caltech", "dslr", "webcam"]}
            num_domains = len(domains[self.cfg.DATASET.NAME])
            num_clients_per_domain = self.cfg.DATASET.USERS // num_domains

            print("Test acc of clients:", test_acc)

            for i in range(num_domains):
                start_idx = i * num_clients_per_domain
                end_idx = (i + 1) * num_clients_per_domain

                if client_ids is not None:
                    domain_clients = []
                    for idx, acc_val in enumerate(test_acc):
                        client_idx = client_ids[idx] if idx < len(client_ids) else None
                        if client_idx is not None and start_idx <= client_idx < end_idx:
                            domain_clients.append(acc_val)
                    accs = domain_clients
                else:
                    accs = test_acc[start_idx:end_idx]

                if accs and len(accs) > 0:
                    domain_mean = np.mean(accs)
                    domain_std = np.std(accs) if len(accs) > 1 else 0.0
                    print("Test acc of", domains[self.cfg.DATASET.NAME][i], f"{domain_mean:.2f}", "+/-", f"{domain_std:.2f}")
                else:
                    print("Test acc of", domains[self.cfg.DATASET.NAME][i], "N/A (no clients)")

            if test_acc and len(test_acc) > 0:
                print("Test acc of all", f"{np.mean(test_acc):.2f}", f"{np.std(test_acc):.2f}")
            else:
                print("Test acc of all: N/A (no clients)")

        return avg_acc

    def _finalize_training(self) -> None:
        """Finalize training."""
        print("------------Training completed, evaluating all clients-------------")
        all_clients = list(range(0, self.cfg.DATASET.USERS))
        final_avg_acc = self.evaluate_clients(all_clients, self.num_comm_rounds - 1,
                                            is_global=False)

        self.save_model(self.client_weights['local'], "last_model.pt")
        print(f"Saved final model, Round: {self.num_comm_rounds}, Accuracy: {final_avg_acc}")

        self.trainer.fed_after_train()

        generate_performance_table(self.metrics, self.cfg, self.is_special_dataset,
                                 self.best_round, self.best_base_round)

    def _load_saved_model(self):
        """Load saved model."""
        start_time = time.time()

        try:
            saved_checkpoint = torch.load(self.args.model_dir)
            if isinstance(saved_checkpoint, list):
                print(f"Successfully loaded model with {len(saved_checkpoint)} client weights")
            else:
                print(f"Successfully loaded model, type: {type(saved_checkpoint)}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            return None

        print(f"Model loading completed in {time.time() - start_time:.2f} seconds")
        return saved_checkpoint

    def _distribute_loaded_weights(self, saved_checkpoint):
        """Distribute loaded weights to clients."""
        print("------------Evaluation mode started-------------")
        all_clients = list(range(0, self.cfg.DATASET.USERS))

        for client_id in all_clients:
            self.client_weights['local'][client_id] = saved_checkpoint[client_id]
            print(f"Client {client_id} loaded {len(self.client_weights['local'][client_id])} module parameters")

            if client_id == 0:
                print(f"Client {client_id} detailed parameter info:")
                for param_name, param in self.client_weights['local'][client_id].items():
                    print(f"  - {param_name}: shape={param.shape}")

    def _execute_evaluation(self):
        """Execute evaluation."""
        all_clients = list(range(0, self.cfg.DATASET.USERS))

        if self.cfg.DATASET.SUBSAMPLE_CLASSES == 'new' and not self.is_special_dataset:
            print(f"Testing new classes, performing global evaluation only")
            self.evaluate_clients(all_clients, 0, is_global=True)
        else:
            self.evaluate_clients(all_clients, 0, is_global=False)

            if not self.is_special_dataset:
                self.evaluate_clients(all_clients, 0, is_global=True)

    def _generate_test_report(self):
        """Generate test report."""
        generate_performance_table(self.metrics, self.cfg, self.is_special_dataset,
                                 self.best_round, self.best_base_round, is_test=True)

        print("------------Evaluation mode completed-------------")
