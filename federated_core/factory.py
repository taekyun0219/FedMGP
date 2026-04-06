# federated_core/factory.py

from typing import Type, List
from .base_federated_learner import BaseFederatedLearner
from .trainers.fedpgp_learner import FedPGPLearner
from .trainers.fedmgp_learner import FedMGPLearner
from .trainers.fedmgpv2_learner import FedMGPV2Learner
from .trainers.fedopt_learner import FedOPTLearner
from .trainers.ivlp_learner import IVLPLearner
from .trainers.promptfl_learner import PromptFLLearner
from .trainers.fedtpg_learner import FedTPGLearner
from .trainers.fedmopg_learner import FedMoPGLearner
from .trainers.vpt_learner import VPTLearner
from .trainers.fedcocoop_learner import FedCoCoOpLearner
from .trainers.fedmaple_learner import FedMaPLeLearner
from .trainers.default_learner import DefaultLearner
from .trainers.clip_learner import CLIPLearner

class FederatedLearnerFactory:
    """Factory class for creating federated learner instances."""

    _learner_registry = {
        'FedPGP': FedPGPLearner,
        'FedMGP': FedMGPLearner,
        'FedMGPV2': FedMGPV2Learner,
        'FedOPT': FedOPTLearner,
        'PromptFolio': FedOPTLearner,
        'IVLP': IVLPLearner,
        'PromptFL': PromptFLLearner,
        'FedTPG': FedTPGLearner,
        'VPT': VPTLearner,
        'FedCoCoOp': FedCoCoOpLearner,
        'MaPLe': FedMaPLeLearner,
        'CLIP': CLIPLearner,
        'fedavg': DefaultLearner,
        'fedprox': DefaultLearner,
        'local': DefaultLearner,
    }

    @classmethod
    def create_learner(cls, model_name: str, cfg, args) -> BaseFederatedLearner:
        """Create a federated learner instance based on model name."""
        learner_class = cls._learner_registry.get(model_name)

        if learner_class is None:
            print(f"Warning: No specific implementation for '{model_name}', using default")
            learner_class = DefaultLearner

        print(f"Creating federated learner: {learner_class.__name__} (model: {model_name})")
        return learner_class(cfg, args)

    @classmethod
    def register_learner(cls, model_name: str, learner_class: Type[BaseFederatedLearner]) -> None:
        """Register a new learner type."""
        cls._learner_registry[model_name] = learner_class
        print(f"Registered new learner: {model_name} -> {learner_class.__name__}")

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported model names."""
        return list(cls._learner_registry.keys())

    @classmethod
    def list_supported_models(cls) -> None:
        """Print supported model list."""
        print("Supported federated learning models:")
        for model_name, learner_class in cls._learner_registry.items():
            print(f"  - {model_name}: {learner_class.__name__}")

    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        """Check if a model is supported."""
        return model_name in cls._learner_registry
