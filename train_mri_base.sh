CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=29507 main_train.py --dataset_name mri --sample_n 0 --gen_max_len 256 --gen_min_len 0 --batch_size 4 --epochs 20 --save_dir results/mask_prompt --seed 42 --init_lr 3e-5 --min_lr 3e-9 --warmup_lr 3e-7 --weight_decay 0.005 --warmup_steps 4000 --beam_size 3  --monitor_metric mic_f1


