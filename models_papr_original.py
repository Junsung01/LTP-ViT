import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

# python main.py   --data-path /home/esoc/datasets/ILSVRC2012   --eval  --model vit_base_patch16_224.augreg_in21k_ft_in1k    --input-size 224  --tome_r 0  --test_speed --papr   --papr_ratio 0.9 

def apply_papr(x: torch.Tensor, f: torch.Tensor, z: float) -> torch.Tensor:
    """
    x: input ViT tokens of size (batch, N, c) (including CLS token at index 0)
    f: proposal ConvNet features of size (batch, K, h, w)
    z: keeping ratio for tokens (0.0 ~ 1.0)
    """  
    b, n, c = x.shape
    # n includes CLS token, so patches are n-1
    num_patches = n - 1
    h1 = w1 = int(np.sqrt(num_patches)) # spatial resolution of tokens
    
    # 남길 토큰 수 (CLS 토큰 제외하고 비율 계산)
    nt = int(num_patches * z) 

    # 1. Proposal Feature에서 중요도 맵(Fd) 추출 (Channel Mean)
    # f: [B, C_cnn, H_cnn, W_cnn] -> Fd: [B, H_cnn, W_cnn]
    Fd = f.mean(dim=1) 
    
    # 2. ViT 패치 해상도에 맞춰 보간 (Interpolation) -> Patch Significance Map (P)
    # Fd: [B, H, W] -> unsqueeze -> [B, 1, H, W] -> interpolate -> [B, 1, h1, w1]
    P = F.interpolate(Fd.unsqueeze(1), size=(h1, w1), mode="bicubic", align_corners=False)
    P = P.view(b, -1) # [B, num_patches]

    # 3. 중요도가 높은 패치 인덱스 추출
    # argsort는 오름차순이므로 descending=True
    patch_indices = P.argsort(dim=1, descending=True)[:, :nt] 

    # 4. 인덱스 조정 (CLS 토큰이 0번이므로 +1)
    patch_indices += 1 

    # 5. CLS 토큰 인덱스(0) 추가
    class_indices = torch.zeros(b, 1, dtype=torch.long, device=patch_indices.device)
    
    # M: [B, nt + 1] (CLS + Selected Patches)
    M = torch.cat([class_indices, patch_indices], dim=1)
    
    # 6. Gather를 이용해 토큰 선택
    # M을 확장: [B, nt+1, 1] -> [B, nt+1, C]
    # x: [B, N, C]
    x_pruned = x.gather(dim=1, index=M.unsqueeze(-1).expand(b, -1, c))

    return x_pruned

class PaPrWrapper(nn.Module):
    def __init__(self, vit_model_name, cnn_model_name='mobileone_s0', ratio=0.5, pretrained=True, **kwargs):
        super().__init__()
        self.ratio = ratio
        
        # 1. Main Backbone (ViT)
        print(f"[PaPr] Loading ViT Backbone: {vit_model_name}")
        self.vit = timm.create_model(vit_model_name, pretrained=pretrained, **kwargs)
        
        # 2. Proposal Network (CNN) - 가볍고 빠른 모델 사용
        print(f"[PaPr] Loading Proposal CNN: {cnn_model_name}")
        self.cnn = timm.create_model(cnn_model_name, features_only=True, pretrained=True)
        self.cnn.eval() # CNN은 학습되지 않도록 고정 (선택사항)
        
                # ✨ MobileOne 모델일 경우 Reparameterization 수행 (Inference 속도 향상)
        if 'mobileone' in cnn_model_name:
            print(f"[PaPr] Reparameterizing {cnn_model_name} for inference...")
            # timm의 MobileOne 구현에 따라 reparameterize 호출
            if hasattr(self.cnn, 'reparameterize'):
                self.cnn.reparameterize()
            else:
                # 모델 전체에 메서드가 없는 경우 모듈별로 확인
                for module in self.cnn.modules():
                    if hasattr(module, 'reparameterize'):
                        module.reparameterize()
        
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 1. CNN에서 Feature Map 추출
        # features_only=True이므로 리스트 반환, 보통 마지막이나 중간 feature 사용
        with torch.no_grad():
            cnn_features = self.cnn(x)
            # 가장 마지막 feature map 사용 (해상도가 낮고 정보가 함축됨)
            proposal_map = cnn_features[-1] 

        # 2. ViT Patch Embedding & Positional Embedding
        x_vit = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x_vit.shape[0], -1, -1)
        
        if hasattr(self.vit, 'dist_token') and self.vit.dist_token is not None:
            # DeiT의 경우 Distillation 토큰 처리
            dist_token = self.vit.dist_token.expand(x_vit.shape[0], -1, -1)
            x_vit = torch.cat((cls_token, dist_token, x_vit), dim=1)
        else:
            x_vit = torch.cat((cls_token, x_vit), dim=1)
            
        x_vit = self.vit.pos_drop(x_vit + self.vit.pos_embed)

        # 3. PaPr 적용 (Pruning)
        # ratio가 1.0이면 가지치기 안 함
        if self.ratio < 1.0:
            x_vit = apply_papr(x_vit, proposal_map, self.ratio)

        # 4. ViT Blocks 통과
        x_vit = self.vit.blocks(x_vit)
        x_vit = self.vit.norm(x_vit)

        # 5. Head
        if self.vit.global_pool:
            x_vit = x_vit[:, 1:].mean(dim=1) if self.vit.global_pool == 'avg' else x_vit[:, 0]
        x_vit = self.vit.fc_norm(x_vit)
        x_vit = self.vit.head(x_vit)

        return x_vit

def build_papr_model(vit_name, ratio, **kwargs):
    return PaPrWrapper(vit_name, ratio=ratio, **kwargs)