python train.py  --data_dir data --batch_size 32 --lr 5e-5 --weight_decay 4e-5 --dropout_rate 0.3 --num_layers 1 --num_heads 4 --warm_epochs 1 --patience 5 --lr_scheduler cosine --lr_decay_steps 30  --mode esm --log_dir runs/esm_xiaorong --seed 42 --hid_dim 256 --augment_eps 0.15

