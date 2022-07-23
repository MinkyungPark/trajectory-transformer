# for seed in 42 123
# do
#    for data in "halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" 
#    do
#        python train.py --seed $seed --dataset $data
#    done
# done

# for seed in 42 123
# do
#     for data in "halfcheetah-medium-replay-v2" "hopper-medium-replay-v2" "walker2d-medium-replay-v2" 
#     do
#         python train.py --seed $seed --dataset $data
#     done
# done

# for seed in 42 123
# do
#     for data in "halfcheetah-medium-expert-v2" "hopper-medium-expert-v2" "walker2d-medium-expert-v2" 
#     do
#         python train.py --seed $seed --dataset $data
#     done
# done

for seed in 42 123
do
    for data in "walker2d-medium-expert-v2" 
    do
        python train.py --seed $seed --dataset $data
    done
done


# for seed in 42 123
# do
#     for data in "halfcheetah-expert-v2" "hopper-expert-v2" "walker2d-expert-v2" 
#     do
#         python train.py --seed $seed --dataset $data
#     done
# done


# plan
# python plan.py --dataset "halfcheetah-medium-v2" --horizon 5 --beam_width 32 --loadpath "logs" --model_epoch 48
# python plan.py --dataset "hopper-medium-v2" --horizon 5 --beam_width 32 --loadpath "logs" --model_epoch 48


# for data in "halfcheetah-medium-v2" "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" 
# do
#     python plan.py --dataset $data --horizon 5 --beam_width 32 --loadpath "logs" --model_epoch 48
# done
