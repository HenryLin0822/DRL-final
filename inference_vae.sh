#python3 inference_vae.py --checkpoint ./checkpoints/vae/best_model.pt
#python3 inference_vae.py --checkpoint ./checkpoints/vae/best_model.pt --mode reconstruct --programs "move" "turnLeft"
python3 inference_vae.py --checkpoint ./checkpoints/vae/best_model.pt --mode reconstruct --programs "WHILE notFacingNorth DO turnLeft END" "move" "REPEAT 3 TIMES move END"