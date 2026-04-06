from trainers.fedmgp import FedMGP


class FedMGPV2(FedMGP):
    """FedMGP variant for the 2-pair / top-1 aggregation experiment."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FEDMGPV2.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.FEDMGPV2.NUM_PROMPTS_VISION == 2, "FedMGPV2 fixes vision prompt count to 2"
        assert cfg.TRAINER.FEDMGPV2.NUM_PROMPTS_TEXT == 2, "FedMGPV2 fixes text prompt count to 2"
        assert cfg.TRAINER.FEDMGPV2.TOPK == 1, "FedMGPV2 fixes top-k aggregation to 1"
