#!/bin/bash

BASE_DIR="./experiments/ViT_B_Augreg_only_ToMe"
mkdir -p $BASE_DIR

# --prepatch_drop_ratio 값을 0.1, 0.2, 0.3으로 바꿔가며 실험
for number in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 
do
    # 1. 각 실험마다 고유한 결과 디렉토리 설정
    EXP_DIR="$BASE_DIR/Tome_R_${number}"
    mkdir -p $EXP_DIR
    
    echo "========= 실험 시작: ToMe_R Number = $number ========="

    python main.py \
        --model vit_base_patch16_224.augreg_in21k_ft_in1k \
        --data-path /home/esoc/datasets/ILSVRC2012 \
        --eval \
        --input-size 224 \
        --no-model-ema \
        --test_speed \
        --tome_r $number \
        --output_dir $EXP_DIR \
        2>&1 | tee $EXP_DIR/console_log.txt # 3. 모든 콘솔 출력을 이 파일에 저장

done

echo "모든 실험 완료."