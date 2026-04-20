#!/bin/bash

BASE_DIR="./experiments/ViT_B_MAE_pre_prepruning"
mkdir -p $BASE_DIR
# CUDA_VISIBLE_DEVICES=0 python eval_mae_tome.py   --data_path /home/esoc/datasets/ILSVRC2012    --model vit_base_patch16_224   --mae_checkpoint /home/esoc/junsung/GTP-ViT/mae_finetuned_vit_base.pth   --input_size 224     --eval-crop-ratio 0.875  --tome_r 0  --prepatch_token_drop  --prepatch_drop_ratio 0.5
# --prepatch_drop_ratio 값을 0.1, 0.2, 0.3으로 바꿔가며 실험
for ratio in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8
do
    # 1. 각 실험마다 고유한 결과 디렉토리 설정
    EXP_DIR="$BASE_DIR/pruning_ratio_${ratio}"
    mkdir -p $EXP_DIR
    
    echo "========= 실험 시작: Drop Ratio = $ratio ========="
    python eval_mae_tome.py \
        --model vit_base_patch16_224 \
        --data_path /home/esoc/datasets/ILSVRC2012 \
        --prepatch_token_drop \
        --prepatch_drop_ratio $ratio \
        --input_size 224 \
        --mae_checkpoint /home/esoc/junsung/GTP-ViT/mae_finetuned_vit_base.pth \
        --eval-crop-ratio 0.875 \
        --tome_r 0 \
        --output_dir $EXP_DIR \
        2>&1 | tee $EXP_DIR/console_log.txt # 3. 모든 콘솔 출력을 이 파일에 저장

done

echo "모든 실험 완료."