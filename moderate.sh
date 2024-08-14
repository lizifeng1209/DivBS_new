#CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/tinyimagenet_ModerateBS_01.yaml --seed 0;
#
#CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/tinyimagenet_ModerateBS_01_64.yaml --seed 0;
#
#CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/tinyimagenet_ModerateBS_01_128.yaml --seed 0;
#
#CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/tinyimagenet_ModerateBS_01_256.yaml --seed 0;
#
#CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/tinyimagenet_ModerateBS_05.yaml --seed 0;
#
#CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/tinyimagenet_ModerateBS_08.yaml --seed 0;


CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/cifar100_ModerateBS_01.yaml --seed 0;

CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/cifar100_ModerateBS_01_64.yaml --seed 0;

CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/cifar100_ModerateBS_01_128.yaml --seed 0;

CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/cifar100_ModerateBS_02.yaml --seed 0;

CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/cifar100_ModerateBS_05.yaml --seed 0;

CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/moderate/cifar100_ModerateBS_08.yaml --seed 0;