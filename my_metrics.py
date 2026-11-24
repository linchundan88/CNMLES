# -*- coding: utf-8 -*-
"""统计各模型在每个单元（Unit）以及全部单元上的准确率与 95% 置信区间。

置信区间采用 bootstrap 重采样（默认 1000 次，percentile 法）。

数据来源：results 目录下 my_predict.py / my_predict.bat 生成的结果文件，
文件名形如 <原文件名>_<模型名>.xlsx，其中含 predict_answer 与 Correct_Answer 两列。

用法：
    python my_metrics.py                          # 统计 results 下的模型（默认排除 gpt-6.1-sol）
    python my_metrics.py --n-bootstrap 2000       # 改 bootstrap 次数
    python my_metrics.py --results-dir other_dir  # 指定结果目录
    python my_metrics.py --no-plot                # 只出表，不画柱状图
    python my_metrics.py --no-default-exclude     # 让 gpt-6.1-sol 也参与统计

说明：gpt-6.1-sol 已被 gpt-6.2-astra 替代，默认不计算其指标；也可用 --exclude 追加要排除的模型。

结果同时打印到终端，并保存为：
    results/metrics_accuracy.csv   指标表
    results/metrics_accuracy.png   柱状图（图中文字全部为英文；各模型各单元 + 全部单元平均，误差线为 95%CI）
"""

import argparse
import glob
import os
import unicodedata

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'metrics_accuracy.csv')
CHART_FILE = os.path.join(RESULTS_DIR, 'metrics_accuracy.png')

UNIT_COLUMN = 'Unit'
PREDICT_COLUMN = 'predict_answer'
TRUTH_COLUMN = 'Correct_Answer'
ALL_UNITS_LABEL = '全部单元'

N_BOOTSTRAP = 1000
SEED = 42
ALPHA = 0.05
# gpt-6.1-sol 已被 gpt-6.2-astra 替代，默认不参与性能指标统计
DEFAULT_EXCLUDED_MODELS = ('gpt-6.1-sol',)
REFERENCE_LINE = 60.0  # 柱状图上的红色点线基准线（%）
# 输出（终端表格、CSV、图形）中使用的模型显示名，文件匹配仍用原始模型名
MODEL_DISPLAY_NAMES = {
    'gemini-4-carbon': 'Gemini-4-Carbon',
    'gpt-6.2-astra': 'GPT-6.2 Astra',
    'qwen4-max': 'Qwen4-Max',
    'gpt-6.1-sol': 'GPT-6.1 Sol',
}


def display_model_name(model):
    """把原始模型名（取自结果文件名）转换成输出用的显示名。"""
    return MODEL_DISPLAY_NAMES.get(model, model)


def list_result_files(results_dir, excluded=()):
    """返回 [(模型名, 文件路径)]，模型名取自文件名最后一个下划线之后的部分。

    excluded 中的模型不参与统计（默认排除 gpt-6.1-sol：它已被 gpt-6.2-astra 替代）。
    """
    excluded = set(excluded)
    files = []
    for path in glob.glob(os.path.join(results_dir, '*.xlsx')):
        name = os.path.basename(path)
        if name.startswith('~$'):
            continue  # Excel 打开时产生的临时锁文件
        stem = os.path.splitext(name)[0]
        if '_' not in stem:
            continue
        model = stem.rsplit('_', 1)[1]
        if model in excluded:
            continue
        files.append((model, path))
    return sorted(files)


def bootstrap_ci(correct, rng, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA):
    """bootstrap 重采样估计准确率的 95% 置信区间（percentile 法）。"""
    values = np.asarray(correct, dtype=float)
    n = values.size
    if n == 0:
        return float('nan'), float('nan')

    resampled_means = values[rng.integers(0, n, size=(n_bootstrap, n))].mean(axis=1)
    lower, upper = np.percentile(resampled_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lower), float(upper)


def load_correctness(path):
    """读取结果文件，返回含 Unit 与 is_correct 两列的数据框。"""
    df = pd.read_excel(path, usecols=[UNIT_COLUMN, PREDICT_COLUMN, TRUTH_COLUMN])
    predicted = df[PREDICT_COLUMN].astype(str).str.strip().str.upper()
    truth = df[TRUTH_COLUMN].astype(str).str.strip().str.upper()
    return pd.DataFrame({
        UNIT_COLUMN: df[UNIT_COLUMN],
        'is_correct': (predicted == truth).astype(int),
    })


def unit_sort_key(unit):
    """单元按数值升序排列，非数值单元排在后面。"""
    try:
        return (0, float(unit))
    except (TypeError, ValueError):
        return (1, str(unit))


def summarize(model, unit_label, correct, seed, n_bootstrap, alpha):
    n = len(correct)
    hits = int(np.sum(correct))
    accuracy = hits / n if n else float('nan')
    # 每个单元用独立同种子的随机数发生器，结果不随单元的统计顺序变化
    lower, upper = bootstrap_ci(correct, np.random.default_rng(seed), n_bootstrap, alpha)
    return {
        '模型': model,
        '单元': unit_label,
        '题数': n,
        '正确数': hits,
        '准确率(%)': round(accuracy * 100, 2),
        '95%CI下限(%)': round(lower * 100, 2),
        '95%CI上限(%)': round(upper * 100, 2),
    }


def compute_metrics(results_dir, n_bootstrap=N_BOOTSTRAP, seed=SEED, alpha=ALPHA, excluded=()):
    files = list_result_files(results_dir, excluded)
    if not files:
        raise FileNotFoundError(f'{results_dir} 下没有找到 <文件名>_<模型名>.xlsx 形式的结果文件')

    rows = []
    used_files = []
    for model, path in files:
        df = load_correctness(path)
        display_model = display_model_name(model)
        used_files.append((display_model, path, len(df)))

        units = sorted(df[UNIT_COLUMN].unique().tolist(), key=unit_sort_key)
        for unit in units:
            values = df.loc[df[UNIT_COLUMN] == unit, 'is_correct'].to_numpy()
            rows.append(summarize(display_model, unit, values, seed, n_bootstrap, alpha))
        rows.append(summarize(display_model, ALL_UNITS_LABEL, df['is_correct'].to_numpy(),
                              seed, n_bootstrap, alpha))

    return pd.DataFrame(rows), used_files


def display_width(text):
    """按终端显示宽度计算字符串宽度（中日韩字符按 2 个字符宽计）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1 for ch in str(text))


def pad(text, width, align='left'):
    """按显示宽度补空格对齐。"""
    text = str(text)
    gap = max(width - display_width(text), 0)
    return text + ' ' * gap if align == 'left' else ' ' * gap + text


def format_table(metrics):
    """把指标表格式化为等宽文本，便于终端阅读。"""
    widths = (16, 10, 6, 8, 10, 22)
    aligns = ('left', 'left', 'right', 'right', 'right', 'right')
    headers = ('模型', '单元', '题数', '正确数', '准确率', '95%CI')

    def render(cells):
        return '  '.join(pad(cell, width, align)
                         for cell, width, align in zip(cells, widths, aligns)).rstrip()

    lines = [render(headers), '-' * 76]
    previous_model = None
    for _, row in metrics.iterrows():
        model = row['模型'] if row['模型'] != previous_model else ''
        previous_model = row['模型']
        lines.append(render((
            model,
            row['单元'],
            int(row['题数']),
            int(row['正确数']),
            '{:.2f}%'.format(row['准确率(%)']),
            '[{:.2f}%, {:.2f}%]'.format(row['95%CI下限(%)'], row['95%CI上限(%)']),
        )))
    return '\n'.join(lines)


def plot_accuracy_bars(metrics, output_file):
    """绘制分组柱状图（图中文字全部为英文）：x 轴为各单元（含 All units），y 轴为准确率，误差线为 95%CI。"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无界面环境也能保存图片
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f'绘制柱状图需要 matplotlib，当前环境不可用：{exc}') from exc

    plt.rcParams['font.family'] = 'DejaVu Sans'

    model_order = list(dict.fromkeys(metrics['模型'].tolist()))
    unit_order = list(dict.fromkeys(metrics['单元'].tolist()))
    lookup = metrics.set_index(['模型', '单元'])

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
    group_count = len(unit_order)
    model_count = len(model_order)
    group_positions = np.arange(group_count)
    bar_width = 0.8 / model_count

    figure, axes = plt.subplots(figsize=(max(9.0, 1.6 * group_count + 3.0), 5.6))

    for j, model in enumerate(model_order):
        offset = (j - (model_count - 1) / 2) * bar_width
        values, lower_errors, upper_errors = [], [], []
        for unit in unit_order:
            row = lookup.loc[(model, unit)]
            accuracy = float(row['准确率(%)'])
            values.append(accuracy)
            lower_errors.append(max(accuracy - float(row['95%CI下限(%)']), 0.0))
            upper_errors.append(max(float(row['95%CI上限(%)']) - accuracy, 0.0))

        positions = group_positions + offset
        bars = axes.bar(positions, values, width=bar_width * 0.92,
                        color=colors[j % len(colors)], label=model,
                        yerr=[lower_errors, upper_errors], capsize=3,
                        error_kw={'elinewidth': 1.0, 'ecolor': '#444444'})

        for bar, value, upper in zip(bars, values, upper_errors):
            axes.text(bar.get_x() + bar.get_width() / 2, value + upper + 0.8,
                      f'{value:.2f}', ha='center', va='bottom', fontsize=8, color='#222222')

    axes.set_xticks(group_positions)
    axes.set_xticklabels(['All units' if unit == ALL_UNITS_LABEL else f'Unit {unit}'
                          for unit in unit_order])
    axes.set_ylabel('Accuracy (%)')
    axes.set_ylim(0, 105)
    axes.set_yticks(np.arange(0, 101, 10))
    axes.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#CCCCCC')
    axes.set_axisbelow(True)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    # 60% 红色点线基准线（画在柱子上层，保证可见）
    axes.axhline(REFERENCE_LINE, color='red', linestyle=':', linewidth=1.6, zorder=4,
                    label='Passing line')
    axes.text(group_count - 0.5, REFERENCE_LINE + 0.8, f'{REFERENCE_LINE:.0f}%',
              ha='right', va='bottom', fontsize=8, color='red', zorder=5)
    axes.set_title('Accuracy of each model per unit and overall\n'
                   '(error bars: 95% bootstrap confidence interval)',
                   fontsize=12, pad=12)
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                ncol=min(model_count + 1, 5), frameon=False)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(figure)
    return output_file


def main():
    parser = argparse.ArgumentParser(description='统计各模型在各单元及全部单元上的准确率与 95% 置信区间')
    parser.add_argument('--results-dir', default=RESULTS_DIR, help='结果文件目录')
    parser.add_argument('--output', default=OUTPUT_FILE, help='指标 CSV 输出路径')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP, help='bootstrap 重采样次数')
    parser.add_argument('--seed', type=int, default=SEED, help='bootstrap 随机种子')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='显著性水平（0.05 对应 95% 置信区间）')
    parser.add_argument('--plot', default=CHART_FILE, help='柱状图输出路径（PNG）')
    parser.add_argument('--no-plot', action='store_true', help='不绘制柱状图')
    parser.add_argument('--exclude', action='append', default=[],
                        help='不参与统计的模型名，可重复指定；默认已排除 gpt-6.1-sol')
    parser.add_argument('--no-default-exclude', action='store_true',
                        help='不使用默认排除列表（即让 gpt-6.1-sol 也参与统计）')
    args = parser.parse_args()

    excluded = [] if args.no_default_exclude else list(DEFAULT_EXCLUDED_MODELS)
    excluded += args.exclude or []

    metrics, used_files = compute_metrics(args.results_dir, args.n_bootstrap, args.seed,
                                          args.alpha, excluded)

    print('使用的结果文件：')
    for model, path, n in used_files:
        print(f'  {model}: {path}（{n} 题）')
    if excluded:
        print(f'已排除模型（不参与统计）：{", ".join(sorted(set(excluded)))}')
    print(f'\n置信区间：bootstrap {args.n_bootstrap} 次，percentile 法，'
          f'置信水平 {100 * (1 - args.alpha):.0f}%，随机种子 {args.seed}\n')

    print(format_table(metrics))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    metrics.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'\n指标已保存到 {args.output}')

    if not args.no_plot:
        chart_file = plot_accuracy_bars(metrics, args.plot)
        print(f'柱状图已保存到 {chart_file}')


if __name__ == '__main__':
    main()
