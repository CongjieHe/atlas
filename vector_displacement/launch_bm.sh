CUDA_VISIBLE_DEVICES="4,5,6,7"

size=base
DATA_DIR="/mnt/data/hcj/atlas"
MASTER_PORT="29500"
MASTER_ADDR="localhost"

# Prepare train/dev/test data from corpus:
TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2018_sampled/text-list-100-sec_sampled.jsonl"
INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2018_sampled/infobox_sampled.jsonl"

TRAIN_FILES="${TEXTS}.shuf.train ${INFOBOXES}.shuf.train"
EVAL_FILES="${TEXTS}.shuf.valid ${INFOBOXES}.shuf.valid ${TEXTS}.shuf.test ${INFOBOXES}.shuf.test"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=${size}-wiki-mlm-pretrain_sampled
PRECISION="fp16" # "bf16"


torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    train.py \
    --retrieve_with_rerank --n_to_rerank_with_retrieve_with_rerank 100 \
    --train_retriever --gold_score_mode "pdist" \
    --use_gradient_checkpoint_reader --use_gradient_checkpoint_retriever \
    --shard_grads --shard_optim \
    --precision ${PRECISION} \
    --temperature_gold 0.01 --temperature_score 0.01 \
    --refresh_index 100 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --passages ${TRAIN_FILES} \
    --target_maxlength 64 \
    --dropout 0.1 \
    --lr 1e-4 --lr_retriever 1e-5\
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 384 \
    --model_path none \
    --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 512 \
    --n_context 20 --retriever_n_context 20 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${DATA_DIR}/checkpoints \
    --save_freq 100 --eval_freq 200 --log_freq 100 \
    --total_steps 1200 \
    --warmup_steps 100 \
    --min_words_per_lm_instance 10 \
    --task "lm" \
    --min_lm_context_ratio 0.25 --max_lm_context_ratio 0.75 \
    --save_index_path ${SAVE_DIR}${EXPERIMENT_NAME}/index \
    --per_gpu_embedder_batch_size 1024 \
