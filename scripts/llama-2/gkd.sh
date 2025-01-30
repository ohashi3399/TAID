python train.py \
    --student_model EleutherAI/pythia-410m \
    --teacher_model stabilityai/stablelm-zephyr-3b \
    --data_path data/stablelm \
    --output_dir logs/stablelm-gkd \
    --batch_size 8 \
    --num_epochs 5 \
    --loss_type js \
    --js_beta 0.1 \
    --sampling_type mixed 
