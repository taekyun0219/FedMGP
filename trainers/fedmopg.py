import os.path as osp

from einops import repeat
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Reuse FedTPG's CLIP path because it already exposes prompt injection hooks
    # in both the text and vision transformers.
    design_details = {
        "trainer": "FedTPG",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }

    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x_q, x_kv=None, **kwargs):
        x_q = self.norm(x_q)

        if exists(x_kv):
            x_kv = self.norm_context(x_kv)
        else:
            x_kv = x_q

        return self.fn(x_q, x_kv, x_kv, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, latent_dim, kv_dim, cross_heads=4, seq_dropout_prob=0.0):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob
        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    nn.MultiheadAttention(
                        latent_dim,
                        num_heads=cross_heads,
                        kdim=kv_dim,
                        vdim=kv_dim,
                        dropout=seq_dropout_prob,
                        batch_first=True,
                    ),
                    context_dim=kv_dim,
                ),
                FeedForward(latent_dim),
            ]
        )

    def forward(self, data, soft_prompt, mask=None):
        b = data.shape[0]
        x = repeat(soft_prompt, "n d -> b n d", b=b)
        cross_attn, cross_ff = self.cross_attend_blocks
        x, _ = cross_attn(x, data, key_padding_mask=mask)
        x = cross_ff(x) + x
        return x


class SelfAttention(nn.Module):
    def __init__(self, depth, latent_dim, latent_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            latent_dim,
                            nn.MultiheadAttention(
                                latent_dim, num_heads=latent_heads, batch_first=True
                            ),
                        ),
                        FeedForward(latent_dim),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for self_attn, self_ff in self.layers:
            x = self_attn(x, key_padding_mask=mask)[0] + x
            x = self_ff(x) + x
        return x


class MixturePromptGenerator(nn.Module):
    def __init__(
        self,
        num_prompt_pairs,
        prompt_len,
        prompt_depth,
        prompt_dim=512,
        depth=0,
        self_heads=4,
        cross_heads=4,
        textemb_dim=512,
    ):
        super().__init__()
        self.num_prompt_pairs = num_prompt_pairs
        self.prompt_len = prompt_len
        self.prompt_depth = prompt_depth

        total_prompt_tokens = num_prompt_pairs * prompt_depth * prompt_len
        soft_prompt = torch.empty(total_prompt_tokens, prompt_dim)
        nn.init.normal_(soft_prompt, std=0.02)
        self.soft_prompt = nn.Parameter(soft_prompt)

        self.encoder = CrossAttention(
            latent_dim=prompt_dim,
            kv_dim=textemb_dim,
            cross_heads=cross_heads,
        )
        self.depth = depth
        if depth > 0:
            self.transformer = SelfAttention(
                depth=depth, latent_dim=prompt_dim, latent_heads=self_heads
            )
        self.text_head = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.vision_head = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, prompt_dim),
        )

    def forward(self, class_token_embeddings, class_token_mask=None):
        prompt_bank = self.encoder(
            class_token_embeddings, self.soft_prompt, mask=class_token_mask
        )
        if self.depth > 0:
            prompt_bank = self.transformer(prompt_bank)

        prompt_bank = prompt_bank.squeeze(0)
        prompt_bank = prompt_bank.reshape(
            self.num_prompt_pairs, self.prompt_depth, self.prompt_len, -1
        )
        text_prompt_bank = self.text_head(prompt_bank)
        vision_prompt_bank = self.vision_head(prompt_bank)

        return text_prompt_bank, vision_prompt_bank


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.conv1
        self.class_embedding = clip_model.class_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_pre = clip_model.ln_pre
        self.transformer = clip_model.transformer
        self.ln_post = clip_model.ln_post
        self.proj = clip_model.proj

    def forward(self, x, vis_ctx=None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x, vis_ctx if vis_ctx is not None else [], False)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, text_ctx):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, text_ctx, True)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(classnames)
        self.n_ctx = cfg.TRAINER.FEDMOPG.N_CTX
        self.ctx_depth = cfg.TRAINER.FEDMOPG.D_CTX
        self.num_prompt_pairs = cfg.TRAINER.FEDMOPG.NUM_PROMPT_PAIRS
        self.prompt_prefix = " ".join(["X"] * self.n_ctx)
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

        prompts = [self.prompt_prefix + " " + name + "." for name in self.classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])

        class_tokens = torch.cat([clip.tokenize(name) for name in self.classnames])
        self.register_buffer("tokenized_classnames", class_tokens)

        self.meta_net = MixturePromptGenerator(
            num_prompt_pairs=self.num_prompt_pairs,
            prompt_len=self.n_ctx,
            prompt_depth=self.ctx_depth,
            prompt_dim=clip_model.ln_final.weight.shape[0],
            depth=cfg.TRAINER.FEDMOPG.DEPTH,
            cross_heads=cfg.TRAINER.FEDMOPG.CROSS_HEADS,
            self_heads=cfg.TRAINER.FEDMOPG.SELF_HEADS,
            textemb_dim=clip_model.ln_final.weight.shape[0],
        )
        self.meta_net.half()

    def get_class_token_embeddings(self):
        tokenized = self.tokenized_classnames
        embeddings = self.token_embedding(tokenized).type(self.dtype)
        pad_mask = tokenized.eq(0)
        embeddings = embeddings.reshape(1, -1, embeddings.shape[-1])
        pad_mask = pad_mask.reshape(1, -1)
        return embeddings, pad_mask

    def construct_prompts(self, ctx, label=None):
        prefix = self.token_prefix if label is None else self.token_prefix[label]
        suffix = self.token_suffix if label is None else self.token_suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        class_token_embeddings, class_token_mask = self.get_class_token_embeddings()

        generated_text_ctx, generated_vis_ctx = self.meta_net(
            class_token_embeddings, class_token_mask
        )

        prompt_groups = []
        for prompt_idx in range(self.num_prompt_pairs):
            text_ctx = generated_text_ctx[prompt_idx]
            vis_ctx = generated_vis_ctx[prompt_idx]

            shallow_text_ctx = text_ctx[0].unsqueeze(0).expand(self.n_cls, -1, -1)
            prompt_vectors = self.construct_prompts(shallow_text_ctx)
            deep_text_ctx = text_ctx[1:] if text_ctx.shape[0] > 1 else text_ctx[:0]

            prompt_groups.append(
                {
                    "prompt_vectors": prompt_vectors,
                    "text_ctx": deep_text_ctx,
                    "vis_ctx": vis_ctx,
                }
            )

        return prompt_groups


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_prompt_pairs = cfg.TRAINER.FEDMOPG.NUM_PROMPT_PAIRS
        self.use_divergent_loss = getattr(
            cfg.TRAINER.FEDMOPG, "USE_DIVERGENT_LOSS", False
        )
        self.divergent_loss_weight = getattr(
            cfg.TRAINER.FEDMOPG, "DIVERGENT_LOSS_WEIGHT", 0.1
        )
        self.divergent_loss_type = getattr(
            cfg.TRAINER.FEDMOPG, "DIVERGENT_LOSS_TYPE", "cos"
        )

    def forward(self, image, label=None):
        prompt_groups = self.prompt_learner()
        image = image.type(self.dtype)

        text_features_list = []
        image_features_list = []
        logits_per_group = []
        logit_scale = self.logit_scale.exp()

        for prompt_group in prompt_groups:
            text_features = self.text_encoder(
                prompt_group["prompt_vectors"],
                self.tokenized_prompts,
                prompt_group["text_ctx"],
            )
            image_features = self.image_encoder(image, prompt_group["vis_ctx"])

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_list.append(text_features)
            image_features_list.append(image_features)

            logits = logit_scale * image_features @ text_features.t()
            logits_per_group.append(logits)

        stacked_logits = torch.stack(logits_per_group, dim=0)
        avg_logits = stacked_logits.mean(dim=0)

        if self.training and label is not None:
            cls_losses = [F.cross_entropy(logits, label) for logits in logits_per_group]
            avg_loss = torch.stack(cls_losses).mean()

            divergent_loss = torch.tensor(0.0, device=image.device, dtype=self.dtype)
            if self.use_divergent_loss and self.num_prompt_pairs > 1:
                text_divergent_loss = self._compute_divergent_loss(
                    text_features_list, batch_dim_first=False
                )
                vision_divergent_loss = self._compute_divergent_loss(
                    image_features_list, batch_dim_first=True
                )
                divergent_loss = (text_divergent_loss + vision_divergent_loss) / 2.0
                divergent_loss = divergent_loss * self.divergent_loss_weight
                if divergent_loss < 0:
                    divergent_loss = torch.abs(divergent_loss)

            total_loss = avg_loss
            if self.use_divergent_loss and divergent_loss > 0:
                total_loss = total_loss + divergent_loss

            if self.use_divergent_loss:
                return total_loss, avg_logits, divergent_loss
            return avg_loss, avg_logits

        return avg_logits

    def _compute_divergent_loss(self, features_list, batch_dim_first=False):
        device = features_list[0].device
        dtype = features_list[0].dtype
        num_prompts = len(features_list)

        if num_prompts <= 1:
            return torch.tensor(0.0, device=device, dtype=dtype)

        first_dim_size = features_list[0].size(0)
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for sample_idx in range(first_dim_size):
            sample_loss = torch.tensor(0.0, device=device, dtype=dtype)
            num_pairs = 0

            for i in range(num_prompts):
                for j in range(i + 1, num_prompts):
                    feat_i = features_list[i][sample_idx]
                    feat_j = features_list[j][sample_idx]

                    if self.divergent_loss_type == "cos":
                        sim = F.cosine_similarity(
                            feat_i.unsqueeze(0), feat_j.unsqueeze(0), dim=1
                        )[0]
                        pair_loss = 1.0 - sim
                    elif self.divergent_loss_type == "l1":
                        pair_loss = -torch.abs(feat_i - feat_j).mean()
                    elif self.divergent_loss_type == "l2":
                        pair_loss = -torch.sqrt(torch.sum((feat_i - feat_j) ** 2) + 1e-8)
                    else:
                        sim = F.cosine_similarity(
                            feat_i.unsqueeze(0), feat_j.unsqueeze(0), dim=1
                        )[0]
                        pair_loss = 1.0 - sim

                    sample_loss = sample_loss + pair_loss
                    num_pairs += 1

            if num_pairs > 0:
                total_loss = total_loss + (sample_loss / num_pairs)

        return total_loss / first_dim_size


class FedMoPG(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FEDMOPG.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FEDMOPG.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP for FedMoPG")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        if cfg.DATASET.NAME == "ImageNet":
            self.device = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.FEDMOPG.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FEDMOPG.PREC

        if prec == "amp":
            with autocast("cuda"):
                output = self.model(image, label)
                loss = output[0]
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, label)
            loss = output[0]
            self.model_backward_and_update(loss)

        def get_loss_value(loss_tensor_or_scalar):
            if hasattr(loss_tensor_or_scalar, "item"):
                return loss_tensor_or_scalar.item()
            return float(loss_tensor_or_scalar)

        loss_summary = {
            "loss": get_loss_value(loss),
            "acc": compute_accuracy(output[1], label)[0].item(),
        }

        if len(output) >= 3:
            loss_summary["divergent_loss"] = get_loss_value(output[2])

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "tokenized_classnames" in state_dict:
                del state_dict["tokenized_classnames"]

            print(f"Loading weights to {name} from {model_path} (epoch = {epoch})")
            self._models[name].load_state_dict(state_dict, strict=False)
