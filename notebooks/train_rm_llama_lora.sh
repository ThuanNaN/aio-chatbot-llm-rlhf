set -x

CUDA_VISIBLE_DEVICES=0 deepspeed --module openrlhf.cli.train_rm \
   --ckpt.output_dir ./checkpoint/Llama-3.2-1B-rm-dpo \
   --ckpt.save_steps -1 \
   --logger.logging_steps 1 \
   --eval.steps -1 \
   \
   --train.batch_size 512 \
   --train.micro_batch_size 64 \
   --train.max_epochs 1 \
   \
   --model.model_name_or_path thuanan/Llama-3.2-1B-Instruct-Chat-sft \
   --model.gradient_checkpointing_enable \
   \
   --ds.value_head_prefix score \
   --ds.param_dtype bf16 \
   --ds.zero_stage 2 \
   --ds.adam_offload \
   --ds.packing_samples \
   \
   --ds.lora.rank 16 \
   --ds.lora.alpha 32 \
   \
   --data.dataset thuanan/Vi-Alpaca-Preference \
   --data.max_len 2048 \
   --data.apply_chat_template \
   --data.chosen_key chosen \
   --data.rejected_key rejected \
   \
   --adam.lr 5e-6