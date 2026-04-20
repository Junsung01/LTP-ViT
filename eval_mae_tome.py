"""
MAE + ToMe + Prepruning 평가 스크립트
"""
import argparse
import json
import os
import time

import torch
import torch.backends.cudnn as cudnn

def _measure_latency(model, batch_size, img_size, device, ntest=100, warmup=10):
    """Measure latency with synthetic data"""
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(ntest):
            model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    elapsed = time.time() - start

    per_batch_ms = (elapsed / ntest) * 1000
    per_image_ms = per_batch_ms / batch_size
    throughput = batch_size * ntest / elapsed

    return {
        'per_batch_ms': per_batch_ms,
        'per_image_ms': per_image_ms,
        'throughput': throughput
    }
from timm.models import create_model
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("Warning: fvcore not available. Install with: pip install fvcore")

try:
    from torchprofile import profile_macs
    HAS_TORCHPROFILE = True
except ImportError:
    HAS_TORCHPROFILE = False
    print("Warning: torchprofile not available. Install with: pip install torchprofile")


def get_macs(model, x=None, img_size=224):
    """main.py와 동일한 방식으로 MACs 측정"""
    model.eval()
    if x is None:
        x = torch.rand(1, 3, img_size, img_size).cuda()
    
    # fvcore 사용 (더 정확함, PaPr 논문과 동일)
    if HAS_FVCORE:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        return fca.total()  # FLOPs 단위로 반환
    
    # fvcore 없으면 torchprofile 사용
    if HAS_TORCHPROFILE:
        macs = profile_macs(model, x)
        return macs
    
    return None

# ToMe 로컬 폴더 사용
import sys
sys.path.insert(0, '/home/esoc/junsung/GTP-ViT/ToMe')
import tome

from util.pos_embed import load_mae_checkpoint
from models_prepruning9_1_MAE import build_token_drop_by_pos_change_9_1_MAE
#===================PAPR 추가 부분===================
try:
    from models_papr_mae import build_papr_model_mae as build_papr_model
except ImportError:
    print("Warning: models_papr_mae.py not found. PaPr will not work.")
    build_papr_model = None
#=====================================================
def get_args_parser():
    parser = argparse.ArgumentParser('MAE + ToMe + Prepruning evaluation', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--backbone_name', default='vit_base_patch16_224', type=str, help='Backbone model name for prepruning')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--data_path', default='/home/esoc/datasets/ILSVRC2012', type=str)
    parser.add_argument('--mae_checkpoint', required=True, type=str, help='Path to MAE checkpoint')
    
    # ToMe arguments
    parser.add_argument('--tome_r', type=int, default=0, help='ToMe reduction per layer (0 to disable)')
    
    # Prepruning arguments
    parser.add_argument('--prepatch_token_drop', action='store_true', help='Enable prepruning')
    parser.add_argument('--prepatch_drop_ratio', type=float, default=0.0, help='Prepruning drop ratio')
    
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help='Crop ratio for evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='', type=str, help='Directory to store evaluation logs/results')
    parser.add_argument('--latency_only', action='store_true', help='Measure latency (batch_size=1) before evaluation and exit')
    
    #✨ [추가] PaPr arguments
    parser.add_argument('--papr', action='store_true', help='Enable PaPr (Patch Pruning)')
    parser.add_argument('--papr_ratio', type=float, default=1.0, help='Keeping ratio for PaPr (e.g., 0.5 keeps 50% tokens)')
    parser.add_argument('--papr_cnn', type=str, default='mobileone_s0', help='CNN backbone for PaPr proposal')
    
    #✨ [추가] GTP-ViT arguments
    parser.add_argument('--gtp_vit', action='store_true', help='Enable GTP-ViT mode')
    parser.add_argument('--selection', default='None', 
                        choices=['CLSAttnMean', 'CLSAttnMax', 'IMGAttnMean', 'IMGAttnMax', 
                                 'DiagAttnMean', 'DiagAttnMax', 'MixedAttnMean', 'MixedAttnMax',
                                 'CosSimMean', 'CosSimMax', 'SumAttnMax', 'Random', 'None'],
                        type=str, help='Token selection method')
    parser.add_argument('--propagation', default='None',
                        choices=['None', 'Mean', 'GraphProp'],
                        type=str, help='Token propagation method')
    parser.add_argument('--graph_type', default='None', 
                        choices=['None', 'Spatial', 'Semantic', 'Mixed'], 
                        type=str, help='Graph type for propagation')
    parser.add_argument('--num_prop', type=int, default=0, help='Number of tokens to propagate')
    parser.add_argument('--num_neighbours', type=int, default=8, help='Number of neighbours for graph')
    parser.add_argument('--sparsity', type=float, default=1.0, help='Token keep ratio (1.0 = keep all)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for token selection')
    parser.add_argument('--token_scale', action='store_true', default=False, help='Enable token scaling')
    
    return parser


def build_dataset(args):
    """Build ImageNet validation dataset"""
    # Evaluation transform
    t = []
    size = int(args.input_size / args.eval_crop_ratio)
    t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(t)
    
    root = args.data_path + '/val'
    dataset = datasets.ImageFolder(root, transform=transform)
    
    return dataset


@torch.no_grad()
def evaluate(data_loader, model, device):
    """Evaluate model on validation set"""
    model.eval()
    
    correct1 = 0
    correct5 = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward
        outputs = model(images)
        
        # Debug: print output shape on first batch
        if batch_idx == 0:
            print(f"Output shape: {outputs.shape}")
        
        # Handle unexpected output shapes
        if outputs.dim() > 2:
            # If output has extra dimensions, take the mean or reshape
            outputs = outputs.reshape(outputs.size(0), -1)
            if batch_idx == 0:
                print(f"Reshaped output to: {outputs.shape}")
        
        # Top-1 accuracy
        _, pred = outputs.topk(1, 1, True, True)
        correct1 += pred.eq(targets.view(-1, 1)).sum().item()
        
        # Top-5 accuracy
        _, pred5 = outputs.topk(min(5, outputs.size(1)), 1, True, True)
        correct5 += pred5.eq(targets.view(-1, 1).expand_as(pred5)).sum().item()
        
        total += targets.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Batch [{batch_idx}/{len(data_loader)}] - '
                  f'Acc@1: {100.*correct1/total:.2f}% - '
                  f'Acc@5: {100.*correct5/total:.2f}%')
    
    elapsed = time.time() - start_time
    
    acc1 = 100. * correct1 / total
    acc5 = 100. * correct5 / total
    throughput = total / elapsed
    
    print(f'\n=== Evaluation Results ===')
    print(f'Accuracy@1: {acc1:.2f}%')
    print(f'Accuracy@5: {acc5:.2f}%')
    print(f'Total images: {total}')
    print(f'Time: {elapsed:.2f}s')
    print(f'Throughput: {throughput:.2f} images/s')
    
    return {'acc1': acc1, 'acc5': acc5, 'throughput': throughput}


def main(args):
    print(args)
    
    device = torch.device(args.device)
    
    # Set seed
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Prepare output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'args_latest.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Build model
    print(f'\n=== Building Model ===')
    
    #✨ [추가] GTP-ViT Mode
    if args.gtp_vit:
        import models_v3  # GTP-ViT 모델 정의
        
        print(f'GTP-ViT mode enabled')
        print(f'Selection: {args.selection}')
        print(f'Propagation: {args.propagation}')
        print(f'Sparsity: {args.sparsity}')
        print(f'Num Prop: {args.num_prop}')
        print(f'Graph Type: {args.graph_type}')
        
        model = create_model(
            'graph_propagation_vit_base_patch16_224_mae',
            pretrained=True,
            num_classes=1000,
            img_size=args.input_size,
            # GTP-ViT specific options
            selection=args.selection,
            propagation=args.propagation,
            num_prop=args.num_prop,
            num_neighbours=args.num_neighbours,
            sparsity=args.sparsity,
            alpha=args.alpha,
            token_scale=args.token_scale,
            graph_type=args.graph_type
        )
        
        # Load MAE checkpoint
        print(f'Loading MAE checkpoint: {args.mae_checkpoint}')
        checkpoint = torch.load(args.mae_checkpoint, map_location='cpu')
                # checkpoint 구조에 따라 state_dict 추출
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 키 이름에서 'module.' 접두사 제거 (DDP로 저장된 경우)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 모델에 로드 (strict=False로 불일치 키 무시)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'Missing keys: {msg.missing_keys}')
        print(f'Unexpected keys: {msg.unexpected_keys}')
        print(f'✅ GTP-ViT model with MAE weights loaded')
        
    #=========================================================================
    # ✨ [추가] PaPr Mode
    elif args.papr:
        if build_papr_model is None:
            raise ImportError("models_papr.py is missing.")
            
        print(f'PaPr mode enabled')
        print(f'ViT Backbone: {args.model}')
        print(f'CNN Backbone: {args.papr_cnn}')
        print(f'Ratio: {args.papr_ratio}')
        
        # Build PaPr wrapper
        model = build_papr_model(
            vit_name=args.model,
            ratio=args.papr_ratio,
            cnn_model_name=args.papr_cnn,
            pretrained=False,
            num_classes=1000,
            img_size=args.input_size,
            global_pool='avg'
        )
        
        # Load MAE checkpoint into the internal ViT backbone
        print(f'Loading MAE checkpoint: {args.mae_checkpoint}')
        load_mae_checkpoint(model.vit, args.mae_checkpoint, verbose=True)
        print(f'✅ MAE model loaded into PaPr backbone')
    #=========================================================================
    
    elif args.prepatch_token_drop:
        # Prepruning mode: use builder function
        print(f'Prepruning mode enabled')
        print(f'Backbone: {args.backbone_name}')
        print(f'Drop ratio: {args.prepatch_drop_ratio}')
        print(f'MAE Checkpoint: {args.mae_checkpoint}')
        
        model = build_token_drop_by_pos_change_9_1_MAE(
            backbone_name=args.backbone_name,
            drop_ratio=args.prepatch_drop_ratio,
            pretrained=False,
            mae_checkpoint=args.mae_checkpoint,
            num_classes=1000,
            img_size=args.input_size
        )
        print(f'✅ MAE model with prepruning loaded')
    else:
        # Baseline mode: no prepruning
        print(f'Baseline mode (no prepruning)')
        print(f'Model: {args.model}')
        print(f'MAE Checkpoint: {args.mae_checkpoint}')
        
        model = create_model(
            args.model,
            pretrained=False,
            global_pool='avg',  # MAE uses global average pooling
            num_classes=1000,
            img_size=args.input_size,
        )
        
        # Load MAE checkpoint
        model = load_mae_checkpoint(model, args.mae_checkpoint, verbose=True)
        print(f'✅ MAE model loaded successfully')
    
    model.to(device)
    
    # Apply ToMe if enabled
    if args.tome_r > 0:
        print(f'\n=== Applying ToMe ===')
        
        # Get the base model (unwrap if prepruning wrapper exists)
        target_model = model.m if args.prepatch_token_drop else model
        
        if isinstance(args.tome_r, int):
            tome_r_list = [args.tome_r] * len(target_model.blocks)
        else:
            tome_r_list = args.tome_r
        
        tome.patch.mae(target_model, prop_attn=False)
        target_model._tome_info["r"] = tome_r_list.copy()
        target_model.r = tome_r_list
        print(f'ToMe enabled: r={tome_r_list[0]} per layer')
        print(f'prop_attn=False (MAE weighted pooling)')
    else:
        print(f'\nToMe disabled')
    
    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters:,}')
    
    gmacs = None
    # Measure GMACs (main.py와 동일한 방식)
    print(f'\n=== Measuring GMACs ===')
    try:
        macs = get_macs(model, img_size=args.input_size)
        if macs is not None:
            gmacs = macs / 1e9
            print(f'GMACs: {gmacs:.2f}')
        else:
            print('GMACs measurement failed: No measurement library available')
    except Exception as e:
        print(f'GMACs measurement failed: {e}')

    latency_stats = None
    if args.batch_size == 1:
        try:
            latency_stats = _measure_latency(model, 1, args.input_size, device)
            print('\nLatency (batch_size=1):')
            print(f"  Batch latency: {latency_stats['per_batch_ms']:.2f} ms")
            print(f"  Per-image latency: {latency_stats['per_image_ms']:.2f} ms")
            print(f"  Synthetic throughput: {latency_stats['throughput']:.2f} img/s")
        except Exception as e:
            print(f'Latency measurement failed: {e}')
    else:
        print('\nLatency measurement skipped (batch_size != 1)')

    test_stats = None
    if args.latency_only:
        print('\nLatency-only mode enabled; skipping dataset evaluation.')
    else:
        # Build dataset
        print(f'\n=== Building Dataset ===')
        dataset_val = build_dataset(args)
        print(f'Validation set: {len(dataset_val)} images')
        
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        # Evaluate
        print(f'\n=== Starting Evaluation ===')
        test_stats = evaluate(data_loader_val, model, device)
    
    # Persist results
    if args.output_dir:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        metrics = {
            'acc1': test_stats['acc1'] if test_stats else None,
            'acc5': test_stats['acc5'] if test_stats else None,
            'throughput': test_stats['throughput'] if test_stats else None,
            'gmacs': gmacs,
            'n_parameters': n_parameters,
            'latency': latency_stats
        }
        summary = {
            'timestamp': timestamp,
            'mode': 'latency_only' if args.latency_only else 'full_eval',
            'metrics': metrics,
            'config': vars(args)
        }
        result_path = os.path.join(args.output_dir, f'results_{timestamp}.json')
        with open(result_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {result_path}")
    
    return test_stats if test_stats is not None else latency_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAE + ToMe evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
