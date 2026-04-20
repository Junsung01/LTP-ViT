import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tome.utils import parse_r

# tome 호환성 추가
# vit_small_patch16_224.augreg_in21k_ft_in1k 
# vit_base_patch16_224.augreg_in21k_ft_in1k 


# CUDA_VISIBLE_DEVICES=0 python eval_mae_tome.py   --data_path /home/esoc/datasets/ILSVRC2012    --model vit_base_patch16_224   --mae_checkpoint /home/esoc/junsung/GTP-ViT_JS/mae_finetuned_vit_base.pth   --input_size 224     --eval-crop-ratio 0.875  --tome_r 0  --prepatch_token_drop  --prepatch_drop_ratio 0.5
# GTP-ViT 
# python eval_mae_tome.py     --data_path /home/esoc/datasets/ILSVRC2012     --mae_checkpoint /home/esoc/junsung/GTP-ViT_JS/mae_finetuned_vit_base.pth     --gtp_vit     --selection MixedAttnMax     --propagation GraphProp     --sparsity 1.0     --num_prop 16

def _keep_indices_by_pos_change(x_patch_embed, pos_embed_patches, drop_ratio, alpha=0.5):
    B, N, D = x_patch_embed.shape
    if drop_ratio <= 0.0:
        return torch.arange(N, device=x_patch_embed.device).unsqueeze(0).expand(B, N)

    k_drop = int(math.floor(N * drop_ratio))
    k_keep = max(1, N - k_drop)

    # 중요도: ||pos||, 중복성: cosine max
    importance = torch.linalg.vector_norm(pos_embed_patches, dim=-1)        # [B, N]
    
    x_n = F.normalize(x_patch_embed, dim=-1, eps=1e-6)
    sims = torch.bmm(x_n, x_n.transpose(1, 2))
    idx = torch.arange(N, device=x_patch_embed.device)
    sims[:, idx, idx] = float('-inf')
    redundancy, _ = sims.max(dim=2)                                         # [B, N]

    def minmax(t):
        t_min = t.min(dim=1, keepdim=True)[0]
        t_max = t.max(dim=1, keepdim=True)[0]
        return (t - t_min) / (t_max - t_min + 1e-6)

    score = alpha * minmax(importance) - (1.0 - alpha) * minmax(redundancy)
    keep_idx = torch.topk(score, k=k_keep, dim=1, largest=True, sorted=False).indices
    return keep_idx


class ViTTokenDropPosChangeWrapper(nn.Module):
    """
    - drop_ratio==0: timm 원본 forward와 완전히 동일하게 동작
    - drop_ratio>0 : patch_embed와 pos add 사이에서만 토큰 선택을 주입.
                     그 외 경로(보간/pos_drop/blocks/norm/forward_head)는 timm 로직 그대로.
    """
    def __init__(self, timm_vit: nn.Module, drop_ratio: float):
        super().__init__()
        self.m = timm_vit
        self.drop_ratio = float(drop_ratio)

        # 편의 포인터
        self.patch_embed = self.m.patch_embed
        self.pos_drop    = self.m.pos_drop
        self.blocks      = self.m.blocks
        self.norm        = self.m.norm

        # timm VisionTransformer는 보통 아래 속성을 가짐
        self.has_cls = hasattr(self.m, "cls_token") and (self.m.cls_token is not None)
        self.cls_token = getattr(self.m, "cls_token", None)
        self.pos_embed = getattr(self.m, "pos_embed", None)

        # ✨ MAE 모델 감지 (global_pool='avg'이고 fc_norm이 있음)
        self.is_mae = (getattr(self.m, 'global_pool', 'token') == 'avg' and 
                       hasattr(self.m, 'fc_norm'))
        if self.is_mae:
            print("  [Wrapper] Detected MAE model (global_pool='avg', fc_norm)")

        # timm의 head 경로는 forward_head로 마무리하는 것이 가장 안전
        # (global_pool, fc_norm, head_drop, distillation 등 내부 분기 유지)
        assert hasattr(self.m, "forward_head"), "timm VisionTransformer여야 합니다."

        # interpolate_pos_encoding이 있으면 그대로 사용
        self.has_interpolate = hasattr(self.m, "interpolate_pos_encoding")

        # ToMe 호환 정보가 있으면 유지
        self.has_tome = hasattr(self.m, "_tome_info")

    def _interpolate_patch_pos(self, x_patch):
        """
        timm의 interpolate_pos_encoding을 그대로 활용하여
        현재 입력 크기에 맞는 '패치용' positional embedding을 얻습니다.
        - 많은 timm 구현에서 interpolate_pos_encoding은 '패치 부분'만 반환합니다.
        - CLS 토큰의 pos는 self.pos_embed[:, :1, :]을 그대로 사용합니다.
        """
        B, N, D = x_patch.shape
        if self.has_interpolate:
            # 일부 구현은 (x)만 넣으면 내부에서 H,W를 복원해서 patch pos만 반환
            patch_pos = self.m.interpolate_pos_encoding(x_patch)  # [1, N, D] 또는 [B, N, D]
            if patch_pos.shape[0] == 1:
                patch_pos = patch_pos.expand(B, -1, -1)
        else:
            # 입력 크기 고정(예: 224) 가정: pos_embed는 [1, 1+N, D] (CLS + patches)
            patch_pos = self.pos_embed[:, 1:1+N, :].expand(B, -1, -1)
        return patch_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ drop_ratio==0이면 timm 원본 forward와 완전히 동일하게 동작
        if self.drop_ratio <= 0:
            return self.m(x)

        B = x.shape[0]

        # 1) patch embed (원본과 동일)
        x_patch = self.patch_embed(x)   # [B, N, D]
        N, D = x_patch.shape[1], x_patch.shape[2]

        # 2) 현재 입력 크기에 맞춘 패치용 positional embedding 확보(원본과 동일)
        patch_pos = self._interpolate_patch_pos(x_patch)  # [B, N, D]

        # 3) ✨ 토큰 선택 인덱스 계산 (패치/패치-pos만으로 결정)
        keep_idx = _keep_indices_by_pos_change(x_patch, patch_pos, self.drop_ratio)
        # 권장: 접근 패턴 안정화
        keep_idx, _ = torch.sort(keep_idx, dim=1)

        # 4) 패치와 pos를 동일 인덱스로 gather (+contiguous)
        x_patch_kept = torch.gather(
            x_patch, dim=1, index=keep_idx.unsqueeze(-1).expand(-1, -1, D)
        ).contiguous()
        pos_kept = torch.gather(
            patch_pos, dim=1, index=keep_idx.unsqueeze(-1).expand(-1, -1, D)
        ).contiguous()

        # 5) CLS 토큰 추가 (원본 순서를 따름: CLS 먼저, 그 다음 patches)
        if self.has_cls:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x_patch_kept], dim=1)  # [B, 1+N_kept, D]
        else:
            x = x_patch_kept
        
        # 6) Positional embedding 추가 (원본 방식: 전체에 한번에 추가)
        # CLS용 pos와 kept patches용 pos를 결합
        if self.has_cls:
            cls_pos = self.pos_embed[:, :1, :].expand(B, -1, -1)
            combined_pos = torch.cat([cls_pos, pos_kept], dim=1)
        else:
            combined_pos = pos_kept
        
        x = x + combined_pos

        # 7) ✅ ToMe 호환: 현재 길이에 맞춘 size를 지정
        #    대부분의 ToMe timm 패치 구현은 _tome_info["size"]의 shape을 [B, N_wo_cls, 1]로 기대합니다.
        if hasattr(self.m, "_tome_info"):
            # r 설정 유지
            self.m._tome_info["r"] = parse_r(len(self.blocks), getattr(self.m, "r", 0))

            # ★ 핵심: CLS 포함 길이로 size 생성
            L_with_cls = x.shape[1]                    # 예: 168
            size_with_cls = torch.ones(B, L_with_cls, 1, device=x.device)
            self.m._tome_info["size"] = size_with_cls  # ← 길이를 L에 맞춤

            # 만약 여전히 mismatch가 난다면 아래 대안을 시도해보세요(한쪽만 사용).
            # alt_size_with_cls = torch.ones(B, N_kept_wo_cls + (1 if self.has_cls else 0), 1, device=x.device)
            # self.m._tome_info["size"] = alt_size_with_cls

            # 일부 구현은 2D 격자 크기를 참조하기도 합니다. 프루닝 후에는 정사각/직사각이 깨지므로
            # 벡터 size만 제공하는 것이 안전합니다. 그래도 필요하면 다음을 None으로 고정:
            # self.m._tome_info["size_2d"] = None

        # 8) 이후 경로는 timm 원본과 동일
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        
        # ✨ MAE 모델은 forward_head 내부에서 norm을 처리함 (fc_norm)
        # 일반 ViT는 norm을 먼저 적용하고 forward_head 호출
        if not self.is_mae:
            x = self.norm(x)

        # 9) timm의 forward_head로 마무리(원본 분기를 100% 따름)
        logits = self.m.forward_head(x, pre_logits=False)
        return logits

def build_token_drop_by_pos_change_9_1_MAE(backbone_name: str, drop_ratio: float, pretrained: bool = True, 
                                           mae_checkpoint: str = None, **kwargs):
    """
    timm 모델을 생성하고 'By Position Change' Wrapper로 감싸는 빌더 함수
    MAE checkpoint 로딩 지원
    
    Args:
        backbone_name: timm 모델 이름 (e.g., 'vit_base_patch16_224')
        drop_ratio: 토큰 드롭 비율 (0~1)
        pretrained: timm pretrained weights 사용 여부 (mae_checkpoint 사용 시 무시됨)
        mae_checkpoint: MAE checkpoint 경로 (선택적, 지정 시 이 체크포인트로 로드)
        **kwargs: timm.create_model에 전달할 추가 인자
    """
    backbone_lower = backbone_name.lower()
    
    # ✨ MAE checkpoint 사용 모드
    if mae_checkpoint is not None:
        print(f"[MAE Checkpoint Mode]")
        print(f"  Backbone: {backbone_name}")
        print(f"  Checkpoint: {mae_checkpoint}")
        print(f"  Drop ratio: {drop_ratio}")
        
        # ✨ MAE fine-tuned 모델은 global_pool='avg' 사용
        # kwargs에 global_pool이 없으면 추가
        if 'global_pool' not in kwargs:
            kwargs['global_pool'] = 'avg'
            print(f"  Setting global_pool='avg' for MAE fine-tuned model")
        
        # timm 모델 생성 (pretrained=False, MAE 체크포인트로 덮어씀)
        base_model = timm.create_model(backbone_name, pretrained=False, **kwargs)
        
        # MAE checkpoint 로드
        from util.pos_embed import load_mae_checkpoint
        base_model = load_mae_checkpoint(base_model, mae_checkpoint, verbose=True)
        
        print(f"✅ Model '{backbone_name}' loaded with MAE checkpoint")
        
    # Hugging Face MAE 모델 (레거시, 사용 안 함)
    elif 'mae' in backbone_lower and 'hf' in backbone_lower:
        print(f"[MAE-HF Mode] Using Hugging Face MAE (not recommended)")
        import models_mae_hf
        if 'base' in backbone_lower:
            base_model = models_mae_hf.vit_mae_base_hf(pretrained=pretrained)
        elif 'large' in backbone_lower:
            base_model = models_mae_hf.vit_mae_large_hf(pretrained=pretrained)
        elif 'huge' in backbone_lower:
            base_model = models_mae_hf.vit_mae_huge_hf(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown MAE-HF variant: {backbone_name}")
        
    # 일반 timm 모델
    else:
        print(f"[DEBUG] Using timm.create_model for '{backbone_name}'")
        # Use timm for regular models
        base_model = timm.create_model(backbone_name, pretrained=pretrained, **kwargs)
    
    model = ViTTokenDropPosChangeWrapper(base_model, drop_ratio=drop_ratio)
    print(f"Model '{backbone_name}' wrapped with token dropping (ratio: {drop_ratio}) by position change.")
    return model