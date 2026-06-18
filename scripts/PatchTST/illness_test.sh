if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
pred_len=(24 36 48)
label_len=48
model_name=PatchTST

root_path_name=./dataset/
data_path_name=illness_0.csv
model_id_name=illness
data_name=custom_test

random_seed=2021

test_year=0
model_year=0

use_fake=1
fake_root_path=./dataset/
fake_data_path=(fake_illness_24.csv fake_illness_36.csv fake_illness_48.csv)

for i in 0 1 2; do
    for fake_weight in 0 0.1 0.2 0.3 0.4 0.5; do
        python -u run_longExp.py \
          --random_seed $random_seed \
          --is_training 1 \
          --root_path $root_path_name \
          --data_path $data_path_name \
          --model_id $model_id_name'_'$seq_len'_'$pred_len \
          --model $model_name \
          --model_id_name $model_id_name\
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len ${pred_len[$i]} \
          --label_len $label_len \
          --enc_in 7 \
          --e_layers 3 \
          --n_heads 4 \
          --d_model 16 \
          --d_ff 128 \
          --dropout 0.3\
          --fc_dropout 0.3\
          --head_dropout 0\
          --patch_len 24\
          --stride 2\
          --des 'Exp' \
          --target 'OT' \
          --train_epochs 100\
          --lradj 'constant'\
          --patience 15\
          --test_year $test_year\
          --run_type 2\
          --model_year $model_year\
          --use_fake $use_fake\
          --fake_root_path $fake_root_path\
          --fake_data_path ${fake_data_path[$i]} \
          --fake_weight $fake_weight\
          --step1 0\
          --output_folder './test_results2/'\
          --itr 1 --batch_size 24 --learning_rate 0.0025 2>&1 | tee "logs/LongForecasting/${test_year}_${fake_weight}_${model_name}_${model_id_name}_${seq_len}_${pred_len[$i]}.log"
  done
done