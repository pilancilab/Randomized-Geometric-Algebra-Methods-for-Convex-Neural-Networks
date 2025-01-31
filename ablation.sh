export CUDA_VISIBLE_DEVICES=0

for data_name in Amazon IMDB
do
    for num_layers in 2 3 4
    do
        python main_nn_arch.py --data_path /home/ywang/data/ --data_name $data_name --seed 1 --Epochs 20 --train_choice f1 --Hidden 50 --train_num f3 --shuffle --add_eps --num_layers $num_layers --add_skip --batch_size 20
    done
done