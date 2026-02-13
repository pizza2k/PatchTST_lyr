config="512_192_PatchTST_custom_ftM_sl512_ll192_pl192_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0"
base_path="results"
test_year=2018

csv_path="${base_path}/${config}/daily_mse_${test_year}.csv"
output_folder="${base_path}/${config}"

if [ ! -f "${csv_path}" ]; then
    echo "错误: CSV文件不存在 - ${csv_path}"
    exit 1
fi

python -c "
import sys
import os

# 设置路径
sys.path.append('utils')

# 导入并执行
try:
    from plot import plot_daily_mse
    
    result = plot_daily_mse('${csv_path}', ${test_year}, '${output_folder}')
    
    if result:
        print(f'图表保存到: {result}')
    else:
        print('图表生成失败')

except Exception as e:
    print(f'错误: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
