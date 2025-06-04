python3 inference_ddpg.py \
  --ddpg-checkpoint ./checkpoints/ddpg_harvester_1/best_model.pt \
  --vae-checkpoint ./checkpoints/vae_fixed/latest_model_balanced.pt \
  --task harvester \
  --episodes 10 \
