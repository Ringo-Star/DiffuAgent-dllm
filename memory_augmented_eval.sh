export PROJECT_PATH=/home/u7444045/Research/projects/diffusion_reasoning/DiffuAgent/unified_envs/AgentBoard

cd ${PROJECT_PATH}/agentboard

# ALFWorld, memory-augmented
python eval_modular.py \
  --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/memory.yaml \
  --model qwen3 \
  --tasks alfworld_enhanced \
  --max_num_steps 30 \
  --log_path ${PROJECT_PATH}/outputs/qwen3_memory_alfworld

# BabyAI, memory-augmented
python eval_modular.py \
  --cfg-path ${PROJECT_PATH}/agentboard/configs/experiments/memory.yaml \
  --model qwen3 \
  --tasks babyai_enhanced \
  --max_num_steps 30 \
  --log_path ${PROJECT_PATH}/outputs/qwen3_memory_babyai
