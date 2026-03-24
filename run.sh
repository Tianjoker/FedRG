#!/bin/bash

# 定义要运行的算法列表
algorithms=( "fedelc" "fedprox" "symmetricCE" "fedinit" "coteaching" "scaffold" "fedlesams" "fednoro" "moon" "fedLSR" "fedavg" "tcfnll" )


# 总 GPU 数（A100 八卡）
num_gpus=8

# 每张卡最多跑两个任务
tasks_per_gpu=2

# 最大并发任务数 = GPU 数 × 每卡任务数
max_parallel=$(( num_gpus * tasks_per_gpu ))

# 当前并发任务数
running_jobs=0

for i in "${!algorithms[@]}"; do
    algo="${algorithms[$i]}"
    gpu_id=$(( i % num_gpus ))  # 循环分配 GPU
    echo "🚀 Starting training with algorithm: $algo on GPU $gpu_id"

    # 指定 GPU 运行任务，并将日志输出保存
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py gpu_list=[0] algorithms=$algo &

    ((running_jobs++))

    # 控制总并发数为 16
    if [ "$running_jobs" -ge "$max_parallel" ]; then
        wait
        running_jobs=0
    fi
done

# 等待最后一批任务完成
wait

echo "✅ All training jobs finished."
