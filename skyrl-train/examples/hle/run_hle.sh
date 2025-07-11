set -x

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/gsm8k/run_gsm8k.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

DATA_DIR="$HOME/data/hle"
NUM_GPUS=8
LOGGER="wandb"  # change to "console" to print to stdout
NAME="hle_8B_bsz128"
CKPT_DIR="/mnt/data-gcp/users/howard/ckpt"
EXPORT_DIR="/mnt/data-gcp/users/howard/export/$NAME"

# uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
uv run --isolated --extra vllm -m examples.hle.main_hle \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=2 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=128 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.hf_save_interval=1 \
  trainer.export_path="$EXPORT_DIR" \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=16384 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.top_k=20 \
  generator.sampling_params.min_p=0.0 \
  environment.env_class=hle \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="hyen-hle" \
  trainer.run_name="$NAME" \
  trainer.resume_mode=latest\
  trainer.ckpt_path="$CKPT_DIR/$NAME" \
  $@