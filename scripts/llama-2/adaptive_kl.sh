python train.py \
    --student_model TinyLlama/TinyLlama_v1.1 \
    --teacher_model meta-llama/Llama-2-7b-chat-hf \
    --data_path data/llama-2 \
    --output_dir logs/llama-2-adaptivekl \
    --batch_size 4 \
    --num_epochs 5 \
    --loss_type adaptive_kl \
    --adaptive_kl_threshold 0.5 \
    --accumulate_grad_batches 2