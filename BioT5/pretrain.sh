[ -z "${log_path}" ] && log_path="./logs/pretrain_caduceus_122m_molecule"
[ -z "${model_path}" ] && model_path=""
[ -z "${n_node}" ] && n_node=1
[ -z "${n_gpu_per_node}" ] && n_gpu_per_node=2

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY="986353ee165f067cdb44f4be649d449151ca9ad2"

torchrun --nnodes=${n_node} --nproc_per_node=${n_gpu_per_node} biot5/main.py \
    tokenizer_name=google-bert/bert-base-uncased \
    model.random_init=true \
    model.checkpoint_path=${model_path} \
    molecule_dict=dict/selfies_dict.txt \
    hydra.run.dir=${log_path} \
    optim.total_steps=350000 optim.warmup_steps=1000 optim.name=adamw \
    optim.lr_scheduler=cosine optim.base_lr=6e-4 \
    seed=42 \
    model.compile=false \
    pred.every_steps=100 logging.every_steps=2 \
    optim.batch_size=1024 optim.grad_acc=8 \
    output_dir=/raid_elmo/home/lr/zym/MolMamba/pretrain_caduceus_122m_molecule \
    