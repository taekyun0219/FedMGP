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


def load_clip_to_cpu(cfg):  #CLIP backbone을 CPU로 읽어옴
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        #CLIP 내부 transformer가 prompt injection을 지원하는 경로를 재사용하기 위해서
        #바깥 trainer 이름은 FedMoPG지만, text/vision prompt를 실제 transformer에 꽂는 메커니즘은 FedTPG 구조를 빌려씀
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


class PreNorm(nn.Module):   #attention 전에 query와 context를 LayerNorm 하는 wrapper
    #generator 쪽에서 soft prompt latent와 class-token embedding을 attention하기 전에 분포를 정리해 주는 역할
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
    #입력을 값과 gate로 나눠 비선형성을 더 강하게 줌
    #prompt latent를 더 유연하게 바꾸기 위한 블록
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
    #learnable soft prompt token을 query로 쓰고, class-name token embedding을 key/value로 써서 prompt latent를 생성함
    #즉 “이 client가 어떤 class semantic을 갖는가”를 prompt에 주입하는 단계
    def __init__(self, latent_dim, kv_dim, cross_heads=4, seq_dropout_prob=0.0):
        super().__init__()
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
        batch_size = data.shape[0]
        x = repeat(soft_prompt, "n d -> b n d", b=batch_size)
        cross_attn, cross_ff = self.cross_attend_blocks
        x, _ = cross_attn(x, data, key_padding_mask=mask)
        x = cross_ff(x) + x
        return x


class SelfAttention(nn.Module):
    #cross-attention으로 생성된 prompt latent들끼리 다시 상호작용하게 하는 refinement 블록
    #DEPTH > 0일 때만 활성화 -> 하지만 default는 depth = 0이라 스킵됨
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


class CrossModalCoupling(nn.Module):
    #text prompt latent가 vision latent를 보고, vision latent가 text latent를 다시 보는 co-attention refinement
    #독립 생성된 두 modality prompt가 서로 semantic하게 맞물리도록 정렬해 주는 블록
    def __init__(self, latent_dim, heads=4):
        super().__init__()
        self.text_to_vision = PreNorm(
            latent_dim,
            nn.MultiheadAttention(latent_dim, num_heads=heads, batch_first=True),
            context_dim=latent_dim,
        )
        self.vision_to_text = PreNorm(
            latent_dim,
            nn.MultiheadAttention(latent_dim, num_heads=heads, batch_first=True),
            context_dim=latent_dim,
        )
        self.text_ff = FeedForward(latent_dim)
        self.vision_ff = FeedForward(latent_dim)

    def forward(self, text_latent, vision_latent):
        text_refined = self.text_to_vision(text_latent, vision_latent)[0] + text_latent
        vision_refined = self.vision_to_text(vision_latent, text_latent)[0] + vision_latent
        text_refined = self.text_ff(text_refined) + text_refined
        vision_refined = self.vision_ff(vision_refined) + vision_refined
        return text_refined, vision_refined


class SinglePromptGenerator(nn.Module):
    #현재 FedMoPG의 중심
    #text prompt랑 vision prompt를 따로 들고 있고, 각각 class-token embedding을 조건으로 cross-attention함
    def __init__(
        self,
        prompt_len,
        prompt_depth,
        latent_dim=512,
        text_prompt_dim=512,
        vision_prompt_dim=768,
        visual_proto_dim=512,
        depth=0,
        self_heads=4,
        cross_heads=4,
        textemb_dim=512,
        prototype_weight=0.5,
        prototype_momentum=0.9,
    ):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_depth = prompt_depth
        self.prototype_weight = prototype_weight
        self.prototype_momentum = prototype_momentum

        total_prompt_tokens = prompt_depth * prompt_len     #여기서 prompt token 개수는 prompt_depth * prompt_len
        text_soft_prompt = torch.empty(total_prompt_tokens, latent_dim)
        vision_soft_prompt = torch.empty(total_prompt_tokens, latent_dim)
        nn.init.normal_(text_soft_prompt, std=0.02)
        nn.init.normal_(vision_soft_prompt, std=0.02)
        self.text_soft_prompt = nn.Parameter(text_soft_prompt)
        self.vision_soft_prompt = nn.Parameter(vision_soft_prompt)

        self.text_encoder = CrossAttention(
            latent_dim=latent_dim,
            kv_dim=textemb_dim,
            cross_heads=cross_heads,
        )
        self.vision_encoder = CrossAttention(
            latent_dim=latent_dim,
            kv_dim=textemb_dim,
            cross_heads=cross_heads,
        )
        self.text_prototype_encoder = CrossAttention(
            latent_dim=latent_dim,
            kv_dim=visual_proto_dim,
            cross_heads=cross_heads,
        )
        self.vision_prototype_encoder = CrossAttention(
            latent_dim=latent_dim,
            kv_dim=visual_proto_dim,
            cross_heads=cross_heads,
        )
        self.prototype_gate = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.cross_modal_coupling = CrossModalCoupling(latent_dim, heads=cross_heads)

        self.depth = depth
        if depth > 0:
            self.text_transformer = SelfAttention(
                depth=depth, latent_dim=latent_dim, latent_heads=self_heads
            )
            self.vision_transformer = SelfAttention(
                depth=depth, latent_dim=latent_dim, latent_heads=self_heads
            )

        #text/vision transformer hidden dim이 다르기 때문에 text_decoder는 text_hidden_dim으로, vision_decoder는 vision_hidden_dim으로 각각 projection함
        self.text_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, text_prompt_dim),
        )
        self.vision_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, vision_prompt_dim),
        )

    def forward(self, class_token_embeddings, visual_prototypes=None, class_token_mask=None):   #class-token embedding을 입력받아 text/vision prompt latent를 각각 생성
        text_ctx = self.text_encoder(
            class_token_embeddings, self.text_soft_prompt, mask=class_token_mask
        )
        vis_ctx = self.vision_encoder(
            class_token_embeddings, self.vision_soft_prompt, mask=class_token_mask
        )

        if visual_prototypes is not None:
            text_proto_ctx = self.text_prototype_encoder(
                visual_prototypes, self.text_soft_prompt
            )
            vis_proto_ctx = self.vision_prototype_encoder(
                visual_prototypes, self.vision_soft_prompt
            )
            text_gate = self.prototype_gate(text_proto_ctx)
            vis_gate = self.prototype_gate(vis_proto_ctx)
            text_ctx = text_ctx + self.prototype_weight * text_gate * text_proto_ctx
            vis_ctx = vis_ctx + self.prototype_weight * vis_gate * vis_proto_ctx

        text_ctx, vis_ctx = self.cross_modal_coupling(text_ctx, vis_ctx)

        if self.depth > 0:
            text_ctx = self.text_transformer(text_ctx)
            vis_ctx = self.vision_transformer(vis_ctx)

        text_ctx = text_ctx.squeeze(0).reshape(self.prompt_depth, self.prompt_len, -1)  
        vis_ctx = vis_ctx.squeeze(0).reshape(self.prompt_depth, self.prompt_len, -1)

        text_ctx = self.text_decoder(text_ctx)  #text : [prompt_depth, prompt_len, text_dim]
        vis_ctx = self.vision_decoder(vis_ctx)  #vision : [prompt_depth, prompt_len, vision_dim]
        return text_ctx, vis_ctx


class ImageEncoder(nn.Module):
    #CLIP visual encoder wrapper
    #이미지를 patch token sequence로 바꾸고, vision prompt vis_ctx를 transformer에 함께 넣음
    #결과적으로 vision prompt가 patch tokens와 같이 visual transformer 안에서 작동
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
        #이미지를 conv patch embedding으로 바꾸고, class token과 positional embedding을 붙인 뒤, transformer로 보냄
        #여기서 vis_ctx가 실제 vision prompt injection 역할
        #마지막에는 CLS 위치 feature를 꺼내고, 필요하면 proj를 통해 CLIP image embedding 공간으로 투영
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
    #CLIP text encoder wrapper
    #prompt가 삽입된 text token sequence를 CLIP text feature로 바꾸는 역할
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, text_ctx):
        #text prompt가 들어간 prompt sequence에 positional embedding을 더하고 transformer를 통과시킴
        #마지막엔 EOT 위치의 hidden state를 선택해서 text projection을 거쳐 최종 text feature를 만듦
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, text_ctx, True)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    #generator 입력 준비와 prompt 조립을 담당
    #class names를 "X X X class_name." 형태의 base text prompt로 만들어 두고, class name 자체 token도 따로 보관
    #generator는 class-name token만 보고 single text/vision prompt pair를 생성하고, PromptLearner는 그걸 실제 CLIP 입력 형식으로 조합함
    def __init__(self, cfg, classnames, clip_model):
        #여기서 중요한건 text/vision 출력 차원을 CLIP 구조에서 직접 읽음
        super().__init__()
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.n_cls = len(classnames)
        self.n_ctx = cfg.TRAINER.FEDMOPG.N_CTX
        self.ctx_depth = cfg.TRAINER.FEDMOPG.D_CTX
        self.prompt_prefix = " ".join(["X"] * self.n_ctx)
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.text_prompt_dim = clip_model.ln_final.weight.shape[0]
        self.vision_prompt_dim = clip_model.visual.class_embedding.shape[0]
        self.visual_proto_dim = clip_model.visual.output_dim

        prompts = [self.prompt_prefix + " " + name + "." for name in self.classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])

        class_tokens = torch.cat([clip.tokenize(name) for name in self.classnames])
        self.register_buffer("tokenized_classnames", class_tokens)

        self.meta_net = SinglePromptGenerator(
            prompt_len=self.n_ctx,
            prompt_depth=self.ctx_depth,
            latent_dim=self.text_prompt_dim,
            text_prompt_dim=self.text_prompt_dim,
            vision_prompt_dim=self.vision_prompt_dim,
            visual_proto_dim=self.visual_proto_dim,
            depth=cfg.TRAINER.FEDMOPG.DEPTH,
            cross_heads=cfg.TRAINER.FEDMOPG.CROSS_HEADS,
            self_heads=cfg.TRAINER.FEDMOPG.SELF_HEADS,
            textemb_dim=self.text_prompt_dim,
            prototype_weight=cfg.TRAINER.FEDMOPG.PROTOTYPE_WEIGHT,
            prototype_momentum=cfg.TRAINER.FEDMOPG.PROTOTYPE_MOMENTUM,
        )
        self.meta_net.half()
        self.register_buffer(
            "prototype_memory",
            torch.zeros(1, 1, self.visual_proto_dim, dtype=self.dtype),
        )
        self.prototype_initialized = False

    def get_class_token_embeddings(self):
        #class name token들을 CLIP token embedding으로 바꾸고, padding mask를 만듦
        #모든 class token sequence를 하나로 펴서 [1, total_tokens, dim] 형태로 만들기 때문에, 
        #generator는 “이 client가 가진 class semantic 전체”를 한 번에 conditioning signal로 받음
        tokenized = self.tokenized_classnames
        embeddings = self.token_embedding(tokenized).type(self.dtype)
        pad_mask = tokenized.eq(0)
        embeddings = embeddings.reshape(1, -1, embeddings.shape[-1])
        pad_mask = pad_mask.reshape(1, -1)
        return embeddings, pad_mask

    def construct_prompts(self, ctx):
        #text prompt를 실제 입력 sequence로 조립함
        #구조는 [SOS] + generated_text_ctx + suffix
        #여기서 suffix에는 class name과 EOS 이후 토큰이 포함됨
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

    def update_prototype_memory(self, visual_prototypes):
        #client local image feature prototype을 EMA 형태로 유지하여 batch noise를 줄이고 client-level 시각 분포를 보존
        if visual_prototypes is None:
            return

        proto_mean = visual_prototypes.mean(dim=1, keepdim=True).detach().type(self.dtype)
        if not self.prototype_initialized:
            self.prototype_memory.copy_(proto_mean)
            self.prototype_initialized = True
            return

        momentum = self.meta_net.prototype_momentum
        self.prototype_memory.mul_(momentum).add_(proto_mean * (1.0 - momentum))

    def forward(self, visual_prototypes=None):
        #class-token embedding을 준비한 뒤 generator를 호출해서 text prompt와 vision prompt를 한 쌍 생성
        class_token_embeddings, class_token_mask = self.get_class_token_embeddings()
        if visual_prototypes is not None:
            self.update_prototype_memory(visual_prototypes)

        if self.prototype_initialized:
            prototype_context = self.prototype_memory
            if visual_prototypes is not None:
                prototype_context = torch.cat([prototype_context, visual_prototypes], dim=1)
        else:
            prototype_context = visual_prototypes

        text_ctx, vis_ctx = self.meta_net(
            class_token_embeddings,
            visual_prototypes=prototype_context,
            class_token_mask=class_token_mask,
        )

        shallow_text_ctx = text_ctx[0].unsqueeze(0).expand(self.n_cls, -1, -1)
        prompt_vectors = self.construct_prompts(shallow_text_ctx)
        deep_text_ctx = text_ctx[1:] if text_ctx.shape[0] > 1 else text_ctx[:0]

        return {
            "prompt_vectors": prompt_vectors,
            "text_ctx": deep_text_ctx,
            "vis_ctx": vis_ctx,
            "raw_text_ctx": text_ctx,
            "raw_vis_ctx": vis_ctx,
        }


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def build_visual_prototypes(self, image, label=None):
        #client local image feature prototype:
        #frozen image encoder로 뽑은 로컬 이미지 feature를 평균내어 client의 시각 분포를 대표하는 vector sequence를 만듦
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype), None)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if label is None:
            return image_features.mean(dim=0, keepdim=True).unsqueeze(0)

        prototypes = []
        unique_labels = torch.unique(label, sorted=True)
        for cls_id in unique_labels:
            cls_mask = label == cls_id
            prototypes.append(image_features[cls_mask].mean(dim=0))

        if not prototypes:
            return image_features.mean(dim=0, keepdim=True).unsqueeze(0)

        return torch.stack(prototypes, dim=0).unsqueeze(0)

    def forward(self, image, label=None):
        image = image.type(self.dtype)
        visual_prototypes = self.build_visual_prototypes(image, label if self.training else None)
        prompt_group = self.prompt_learner(visual_prototypes=visual_prototypes)
        logit_scale = self.logit_scale.exp()

        text_features = self.text_encoder(
            prompt_group["prompt_vectors"],
            self.tokenized_prompts,
            prompt_group["text_ctx"],
        )
        image_features = self.image_encoder(image, prompt_group["vis_ctx"])

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.training and label is not None:     #학습 중이면 cross_entropy(logits, label)을 계산해서 (loss, logits)를 반환
            loss = F.cross_entropy(logits, label)
            return loss, logits

        return logits


class FedMoPG(TrainerX):    #Dassl trainer interface를 맞추는 클래스
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
        #한 배치 학습 step
        #모델이 반환한 (loss, logits) 중 loss로 backward하고, logits으로 accuracy를 계산함
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

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output[1], label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        #batch dict에서 img, label을 꺼내 현재 device로 옮기는 단순 함수
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        #checkpoint를 불러옴
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
