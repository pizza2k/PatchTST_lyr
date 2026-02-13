import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_daily_mse(csv_path, test_year, output_folder=None):
    if output_folder is None:
        output_folder = os.path.dirname(csv_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   

    # 读取CSV文件
    try:
        mse_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    # 确保日期列是datetime类型
    mse_df['Date'] = pd.to_datetime(mse_df['Date'])
    
    # 按日期排序（确保顺序正确）
    mse_df = mse_df.sort_values('Date')
    
    # 创建图形
    plt.figure(figsize=(20, 8))  # 增加宽度避免拥挤
    
    # 绘制折线图
    plt.plot(mse_df['Date'], mse_df['MSE'], marker='o', linewidth=1.5, markersize=3, alpha=0.8)
    plt.title(f'Daily MSE for {test_year} Predictions', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    
    # 智能调整X轴标签（根据数据量）
    if len(mse_df) > 30:  # 数据点多时减少标签数量
        # 方法1：根据数据量动态决定显示多少个标签
        if len(mse_df) > 100:  # 数据量很大（如全年）
            # 只显示每个月的第一天
            month_starts = mse_df[mse_df['Date'].dt.is_month_start]
            if len(month_starts) > 0:
                plt.xticks(ticks=month_starts['Date'], 
                          labels=month_starts['Date'].dt.strftime('%b\n%Y'), 
                          rotation=0, fontsize=9)
            else:
                # 如果找不到月初数据，均匀取样
                step = max(1, len(mse_df) // 12)  # 显示约12个标签
                xticks_positions = mse_df['Date'][::step]
                plt.xticks(ticks=xticks_positions, 
                          labels=xticks_positions.dt.strftime('%m-%d'), 
                          rotation=45, ha='right', fontsize=9)
        else:
            # 数据量中等，均匀取样显示标签
            step = max(1, len(mse_df) // 15)  # 显示约15个标签
            xticks_positions = mse_df['Date'][::step]
            plt.xticks(ticks=xticks_positions, 
                      labels=xticks_positions.dt.strftime('%m-%d'), 
                      rotation=45, ha='right', fontsize=9)
    else:
        # 数据点少，显示所有标签
        plt.xticks(ticks=mse_df['Date'], 
                  labels=mse_df['Date'].dt.strftime('%Y-%m-%d'), 
                  rotation=45, ha='right', fontsize=10)
    
    # 添加网格和布局调整
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plot_filename = f'daily_mse_{test_year}.png'
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"MSE plot saved to: {plot_path}")
    plt.show()
    
    return plot_path