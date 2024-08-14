#CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/cifar100_DivBS_01.yaml --seed 0;
#
#CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/cifar100_full.yaml --seed 0;
#
#CUDA_VISIBLE_DEVICES=0,1 python main.py --cfg cfg/cifar100_uniform_01.yaml --seed 0;

CUDA_VISIBLE_DEVICES=3,4 python main.py --cfg cfg/test/tinyimagenet_DivBS_01.yaml --seed 0;

CUDA_VISIBLE_DEVICES=3,4 python main.py --cfg cfg/test/tinyimagenet_DivBS_01_3.yaml --seed 0;

CUDA_VISIBLE_DEVICES=3,4 python main.py --cfg cfg/test/tinyimagenet_DivBS_01_5.yaml --seed 0;

CUDA_VISIBLE_DEVICES=3,4 python main.py --cfg cfg/test/tinyimagenet_full.yaml --seed 0;

CUDA_VISIBLE_DEVICES=3,4 python main.py --cfg cfg/test/tinyimagenet_full_3.yaml --seed 0;

CUDA_VISIBLE_DEVICES=3,4 python main.py --cfg cfg/test/tinyimagenet_full_5.yaml --seed 0;

CUDA_VISIBLE_DEVICES=3,4 python main.py --cfg cfg/test/tinyimagenet_uniform_01.yaml --seed 0;

CUDA_VISIBLE_DEVICES=3,4 python main.py --cfg cfg/test/tinyimagenet_uniform_01_3.yaml --seed 0;

CUDA_VISIBLE_DEVICES=3,4 python main.py --cfg cfg/test/tinyimagenet_uniform_01_5.yaml --seed 0;