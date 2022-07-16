for seed in 42 123 312
do
    python train.py --seed $seed --dataset "halfcheetah-medium-v2"
done

for data in "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" 
do
    python train.py --seed 42 --dataset $data
done

for seed in 42
do
    python train.py --seed $seed --dataset "walker2d-medium-v2"
done

plan
python plan.py --dataset "halfcheetah-medium-v2" --horizon 5 --beam_width 32 --loadpath "logs" --model_epoch 48
python plan.py --dataset "hopper-medium-v2" --horizon 5 --beam_width 32 --loadpath "logs" --model_epoch 48


for data in "halfcheetah-medium-v2" "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" 
do
    python plan.py --dataset $data --horizon 5 --beam_width 32 --loadpath "logs" --model_epoch 48
done
