
python trainval.py --name LEVIR-ChangeDINO --gpu_ids 0 --dataset LEVIR-CD --batch_size 16 --fpn_channels 128 --num_epochs 100 --lr 5e-4 
python trainval.py --name WHU-ChangeDINO --gpu_ids 0 --dataset WHU-CD --batch_size 16 --fpn_channels 128 --num_epochs 100 --lr 1e-4
python trainval.py --name SYSU-ChangeDINO --gpu_ids 0 --dataset SYSU-CD --batch_size 16 --fpn_channels 128 --num_epochs 50 --lr 5e-4 
python trainval.py --name S2Looking-ChangeDINO --gpu_ids 0 --dataset S2Looking-CD --batch_size 16 --fpn_channels 128 --num_epochs 50 --lr 5e-4 
python trainval.py --name S1GFloods-ChangeDINO --gpu_ids 0 --dataset S1GFloods_CD_DINO --dataroot ../datasets --dataset_mode sar --stats_file ../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json --batch_size 8 --fpn_channels 128 --num_epochs 100 --lr 1e-4
