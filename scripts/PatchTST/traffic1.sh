if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
label_len=192
model_name=PatchTST

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021

test_year=0
start=16054
end=17517

for pred_len in 96 192 336 720
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
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 64 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --test_year $test_year\
      --run_type 2\
      --model_year 2014\
      --step1 1\
      --test_start $start\
      --test_end $end\
      --output_folder './test_results1/'\
      --itr 1 --batch_size 8 --learning_rate 0.0001 2>&1 | tee "logs/LongForecasting/${test_year}_${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
done