[ -z "${task}" ] && task=molnet
[ -z "${model_path}" ] && model_path="/raid_elmo/home/lr/zym/MolMamba/caduceus-73M/checkpoint-pt-80000"
[ -z "${log_path}" ] && log_path="./logs/finetune_molnet_clintox_ct_tox"
[ -z "${task_dir}" ] && task_dir="/raid_elmo/home/lr/zym/MolMamba/data/biot5_data/tasks"
[ -z "${data_dir}" ] && data_dir="/raid_elmo/home/lr/zym/MolMamba/data/biot5_data/splits/molnet/clintox_ct_tox"
[ -z "${n_node}" ] && n_node=1
[ -z "${n_gpu_per_node}" ] && n_gpu_per_node=1

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

python biot5/main.py \
    task=${task} \
    tokenizer_name=google-bert/bert-base-uncased \
    model.random_init=false \
    model.checkpoint_path=${model_path} \
    data.task_dir=${task_dir} \
    data.data_dir=${data_dir} \
    molecule_dict=dict/selfies_dict.txt \
    hydra.run.dir=${log_path} \
    optim.total_steps=50000 optim.warmup_steps=1000 optim.name=adamw \
    optim.lr_scheduler=cosine optim.base_lr=1e-4 \
    seed=42 \
    model.compile=false \
    pred.every_steps=20 logging.every_steps=2 \
    optim.batch_size=1024 optim.grad_acc=2 optim.test_bsz_multi=1 \
    output_dir=/raid_elmo/home/lr/zym/MolMamba/ \