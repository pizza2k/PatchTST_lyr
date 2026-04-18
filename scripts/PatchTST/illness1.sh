if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
label_len=48
model_name=PatchTST

root_path_name=./dataset/
data_path_name=illness.csv
model_id_name=illness
data_name=custom

random_seed=2021

test_year=0
start=0
end=0

for pred_len in 24 36 48 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model_id_name $model_id_name\
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
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
      --test_year 0\
      --run_type 2\
      --model_year 0\
      --step1 1\
      --test_start $start\
      --test_end $end\
      --output_folder './test_results1/'\
      --itr 1 --batch_size 64 --learning_rate 0.0025 2>&1 | tee "logs/LongForecasting/${test_year}_${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
done