# for seed in 42 123 312
# for seed in 42
# do
#     python train.py --seed $seed --dataset "halfcheetah-medium-v2" --mode "shuffle"
# done

# for data in "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" 
# do
#     python train.py --seed 42 --dataset $data --mode "reverse"
# done

# for seed in 42
# do
#     python train.py --seed $seed --dataset "walker2d-medium-v2" --mode "shuffle"
# done

# plan
# python plan.py --dataset "halfcheetah-medium-v2" --horizon 5 --beam_width 32 --loadpath "logs" --model_epoch 48 --mode "base"
# python plan.py --dataset "hopper-medium-v2" --horizon 5 --beam_width 32 --loadpath "logs" --model_epoch 48 --mode "base"
# 

# for data in "halfcheetah-medium-v2" "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" 
# for data in "hopper-medium-expert-v2" "hopper-medium-replay-v2"
# do
#     python plan.py --dataset $data --horizon 5 --beam_width 32 --loadpath "logs" --model_epoch 48 --mode "base"
# done

for data in "halfcheetah-medium-v2" "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" 
do
    python plan.py --dataset $data --horizon 8 --beam_width 32 --loadpath "logs" --model_epoch 48 --mode "base"
done

for data in "halfcheetah-medium-v2" "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" 
do
    python plan.py --dataset $data --horizon 8 --beam_width 32 --loadpath "logs" --model_epoch 48 --mode "reverse"
done

for data in "halfcheetah-medium-v2" "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" 
do
    python plan.py --dataset $data --horizon 8 --beam_width 32 --loadpath "logs" --model_epoch 48 --mode "shuffle"
done