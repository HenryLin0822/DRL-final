python3 inference_ddpg.py \
  --ddpg-checkpoint ./checkpoints/ddpg_harvester/best_model.pt \
  --vae-checkpoint ./checkpoints/vae_fixed/best_model.pt \
  --task harvester \
  --episodes 10 \
  --output ./results/harvester_evaluation.json