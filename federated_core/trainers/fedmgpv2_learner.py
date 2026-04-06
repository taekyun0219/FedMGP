from .fedmgp_learner import FedMGPLearner


class FedMGPV2Learner(FedMGPLearner):
    """FedMGP variant with two local prompt groups and top-1 aggregation."""

    def aggregate_models(self, client_ids):
        assert self.cfg.TRAINER.FEDMGP.NUM_PROMPTS_TEXT == 2, "FedMGPV2 expects exactly two text prompts"
        assert self.cfg.TRAINER.FEDMGP.NUM_PROMPTS_VISION == 2, "FedMGPV2 expects exactly two vision prompts"
        assert self.cfg.TRAINER.FEDMGP.TOPK == 1, "FedMGPV2 expects top-k aggregation to be 1"
        return super().aggregate_models(client_ids)
