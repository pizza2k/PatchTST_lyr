if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
pred_len=192
label_len=192
model_name=Transformer

model_id_name=tec
data_name=custom_test

random_seed=2021

test_year=0
model_year=0

use_fake=1
fake_root_path=./dataset/
fake_data_path=ddpm_fake_tail_2014.csv

for fake_weight in 0 0.1 0.2 0.3 0.4 0.5; do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path tec_2014.csv \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --model_id_name $model_id_name\
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --label_len $label_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 27 \
      --dec_in 27 \
      --c_out 27 \
      --des 'Exp' \
      --target '(0.0, 20)' \
      --train_epochs 50\
      --patience 5\
      --test_year $test_year\
      --run_type 2\
      --model_year $model_year\
      --use_fake $use_fake\
      --fake_root_path $fake_root_path\
      --fake_data_path $fake_data_path\
      --fake_weight $fake_weight\
      --step1 0\
      --output_folder './test_results3/'\
      --itr 1 --batch_size 32 --learning_rate 0.0001 2>&1 | tee "logs/LongForecasting/${test_year}_${fake_weight}_${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
done