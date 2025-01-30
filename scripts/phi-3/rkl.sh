python train.py \
    --student_model TinyLlama/TinyLlama_v1.1 \
    --teacher_model microsoft/Phi-3-mini-4k-instruct \
    --data_path data/phi-3 \
    --output_dir logs/phi-3-rkl \
    --batch_size 8 \
    --num_epochs 5 \
    --loss_type rkl