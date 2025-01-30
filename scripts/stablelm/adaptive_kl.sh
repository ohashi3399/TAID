python train.py \
    --student_model EleutherAI/pythia-410m \
    --teacher_model stabilityai/stablelm-zephyr-3b \
    --data_path data/stablelm \
    --output_dir logs/stablelm-adaptivekl \
    --batch_size 4 \
    --num_epochs 5 \
    --loss_type adaptive_kl \
    --adaptive_kl_threshold 0.5 \
    --accumulate_grad_batches 2