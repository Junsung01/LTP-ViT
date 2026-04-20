"""
PaPr (Patch Pruning) for MAE fine-tuned models
MAE 모델은 global_pool='avg'와 fc_norm을 사용함
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

# 사용 예시:
# CUDA_VISIBLE_DEVICES=0 python eval_mae_tome.py --data_path /home/esoc/datasets/ILSVRC2012 \
#   --model vit_base_patch16_224 --mae_checkpoint /path/to/mae_finetuned_vit_base.pth \
#   --input_size 224 --papr --papr_ratio 0.5

def apply_papr(x: torch.Tensor, f: torch.Tensor, z: float) -> torch.Tensor:
    """
    x: input ViT tokens of size (batch, N, c) (including CLS token at index 0)
    f: proposal ConvNet features of size (batch, K, h, w)
    z: keeping ratio for tokens (0.0 ~ 1.0)
    
    PaPr 원본 구현과 동일하게 구현
    """  
    b, n, c = x.shape
    # n includes CLS token, so patches are n-1
    num_patches = n - 1
    h1 = w1 = int(np.sqrt(num_patches))  # spatial resolution of tokens
    
    # ✨ PaPr 원본: n (CLS 포함) 기준으로 계산
    nt = int(n * z)

    # 1. Proposal Feature에서 중요도 맵(Fd) 추출 (Channel Mean)
    # f: [B, C_cnn, H_cnn, W_cnn] -> Fd: [B, 1, H_cnn, W_cnn]
    Fd = f.mean(dim=1).unsqueeze(1)
    
    # 2. ViT 패치 해상도에 맞춰 보간 (Interpolation) -> Patch Significance Map (P)
    # ✨ PaPr 원본: align_corners=True
    P = F.interpolate(Fd, size=(h1, w1), mode="bicubic", align_corners=True)
    P = P.view(b, -1)  # [B, num_patches]

    # 3. 중요도가 높은 패치 인덱스 추출
    patch_indices = P.argsort(dim=1, descending=True)[:, :nt]

    # 4. 인덱스 조정 (CLS 토큰이 0번이므로 +1)
    patch_indices += 1

    # 5. CLS 토큰 인덱스(0) 추가
    class_indices = torch.zeros(b, 1, dtype=torch.int64, device=patch_indices.device)
    
    # M: [B, nt + 1] (CLS + Selected Patches)
    M = torch.cat([class_indices, patch_indices], dim=1)
    
    # 6. Gather를 이용해 토큰 선택
    x_pruned = x.gather(dim=1, index=M.unsqueeze(-1).expand(b, -1, c))

    return x_pruned


class PaPrWrapperMAE(nn.Module):
    """
    PaPr Wrapper for MAE fine-tuned models
    MAE 모델은 global_pool='avg'와 fc_norm을 사용
    """
    def __init__(self, vit_model_name, cnn_model_name='mobileone_s0', ratio=0.5, pretrained=True, **kwargs):
        super().__init__()
        self.ratio = ratio
        
        # 1. Main Backbone (ViT) - MAE는 global_pool='avg' 사용
        print(f"[PaPr-MAE] Loading ViT Backbone: {vit_model_name}")
        if 'global_pool' not in kwargs:
            kwargs['global_pool'] = 'avg'
        self.vit = timm.create_model(vit_model_name, pretrained=pretrained, **kwargs)
        
        # 2. Proposal Network (CNN) - 가볍고 빠른 모델 사용
        print(f"[PaPr-MAE] Loading Proposal CNN: {cnn_model_name}")
        self.cnn = timm.create_model(cnn_model_name, features_only=True, pretrained=True)
        self.cnn.eval()  # CNN은 학습되지 않도록 고정
        
        # ✨ MobileOne 모델일 경우 Reparameterization 수행 (Inference 속도 향상)
        if 'mobileone' in cnn_model_name:
            print(f"[PaPr-MAE] Reparameterizing {cnn_model_name} for inference...")
            if hasattr(self.cnn, 'reparameterize'):
                self.cnn.reparameterize()
            else:
                for module in self.cnn.modules():
                    if hasattr(module, 'reparameterize'):
                        module.reparameterize()
        
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 1. CNN에서 Feature Map 추출
        # Note: torch.no_grad() 제거 - requires_grad=False로 이미 gradient 비활성화됨
        # FLOPs 측정 시 정확한 값을 위해 제거
        with torch.no_grad():
            cnn_features = self.cnn(x)
            # 가장 마지막 feature map 사용 (해상도가 낮고 정보가 함축됨)
            proposal_map = cnn_features[-1] 

        # 2. ViT Patch Embedding & Positional Embedding (PaPr 원본과 동일)
        x_vit = self.vit.patch_embed(x)
        x_vit = self.vit._pos_embed(x_vit)  # CLS 추가 + pos_embed + pos_drop 한번에

        # 3. PaPr 적용 (Pruning)
        if self.ratio < 1.0:
            x_vit = apply_papr(x_vit, proposal_map, self.ratio)

        # 4. ViT Blocks 통과
        x_vit = self.vit.blocks(x_vit)

        # 5. MAE 모델: global avg pool (CLS 제외) 후 fc_norm
        x_vit = x_vit[:, 1:, :].mean(dim=1)
        x_vit = self.vit.fc_norm(x_vit)

        # 6. Head
        x_vit = self.vit.head(x_vit)

        return x_vit


def build_papr_model_mae(vit_name, ratio, **kwargs):
    """MAE 모델용 PaPr 빌더"""
    return PaPrWrapperMAE(vit_name, ratio=ratio, **kwargs)
