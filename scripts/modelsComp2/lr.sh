if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
pred_len=192
label_len=192
model_name=PatchTST

root_path_name=./dataset/
data_path_name=tec_2014.csv
model_id_name=tec
data_name=custom_test

random_seed=2021

test_year=0
model_year=0

use_fake=1
fake_root_path=./dataset/models_comp/
fake_data_path=fake_lr_tec_192.csv

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
      --pred_len $pred_len \
      --label_len $label_len \
      --enc_in 27 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --target '(0.0, 20)' \
      --train_epochs 100\
      --patience 15\
      --test_year $test_year\
      --run_type 2\
      --model_year $model_year\
      --use_fake $use_fake\
      --fake_root_path $fake_root_path\
      --fake_data_path $fake_data_path\
      --fake_weight $fake_weight\
      --step1 0\
      --output_folder './test_results_lr/'\
      --itr 1 --batch_size 128 --learning_rate 0.0001 2>&1 | tee "logs/LongForecasting/${test_year}_${fake_weight}_${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
done