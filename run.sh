#!/bin/bash

BASE_DIR="./experiments/ViT_B_Augreg_pre_prepruning"
mkdir -p $BASE_DIR

# --prepatch_drop_ratio 값을 0.1, 0.2, 0.3으로 바꿔가며 실험
for ratio in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8
do
    # 1. 각 실험마다 고유한 결과 디렉토리 설정
    EXP_DIR="$BASE_DIR/pruning_ratio_${ratio}"
    mkdir -p $EXP_DIR
    
    echo "========= 실험 시작: Drop Ratio = $ratio ========="

    python main.py \
        --model vit_base_patch16_224.augreg_in21k_ft_in1k  \
        --data-path /home/esoc/datasets/ILSVRC2012 \
        --eval \
        --prepatch_token_drop \
        --prepatch_backbone vit_base_patch16_224.augreg_in21k_ft_in1k  \
        --prepatch_drop_ratio $ratio \
        --input-size 224 \
        --no-model-ema \
        --test_speed \
        --tome_r 0 \
        --output_dir $EXP_DIR \
        2>&1 | tee $EXP_DIR/console_log.txt # 3. 모든 콘솔 출력을 이 파일에 저장

done

echo "모든 실험 완료."