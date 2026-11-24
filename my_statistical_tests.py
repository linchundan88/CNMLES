# -*- coding: utf-8 -*-
"""对多个模型的预测结果做统计假设检验，判断模型之间是否存在统计学显著差异。

检验方法：
  1. Cochran's Q 检验（多模型整体差异，同时比较 k 个模型）
     - 研究问题：k 个模型在同一批题目上的准确率是否相同（H0：各模型准确率全部相等，
       等价于“每道题上答对的模型身份可随机交换”）。
     - 与 McNemar 的关系：McNemar 检验的多模型推广，k = 2 时与 McNemar 等价。
     - 统计量：Q = (k-1) * (k * ΣC_j² - T²) / (k * ΣR_i - ΣR_i²)，
       其中 C_j 为第 j 个模型答对的题数，R_i 为第 i 题答对的模型数，T 为总答对数。
       H0 成立时 Q 近似服从 χ²(k-1)，df = k - 1，p(卡方) = P(χ²(df) ≥ Q)。
     - 置换检验：题目数有限、且“全对/全错”的题目较多时 χ² 近似可能偏乐观，
       因此补充 Monte Carlo 置换检验——在每道题内随机重排各模型的 0/1 结果
       （H0 下模型可交换，即把该题答对次数随机分配给 k 个模型），重算 Q 得到零分布，
       p(置换) = (1 + 未少于观测 Q 的置换次数) / (B + 1)。
       它不依赖渐近分布。
     - 多重比较：全部单元 + 各单元共（单元数 + 1）次 Q 检验视为同一检验族，
       对置换 p 值做 Holm-Bonferroni 校正以控制族错误率（与分单元 McNemar 的
       “各单元内分别校正”不同，这里跨范围统一校正）；显著性以校正后 p 值为准。
  2. 事后两两 McNemar 精确检验（同一批题目上的配对比较）
     - 用途：整体显著只说明“并非所有模型都相同”，由它定位具体是哪一对模型不同；
       整体不显著时同样报告，供参考。
     - 统计口径：题目逐题配对，只看两个模型结论不一致的题目——
       b = 仅模型 A 答对，c = 仅模型 B 答对；H0：两模型准确率相同（b、c 期望相等）。
     - 精确 p：binomtest(b, b + c, 0.5)，小样本（不一致题数少）时更可靠；
       渐近 p：连续性校正的 McNemar 卡方，( |b - c| - 1 )² / (b + c)。
     - 多重比较：同一范围内所有模型对的精确 p 用 Holm-Bonferroni 逐步校正以控制
       族错误率；全部单元与各单元内部分别校正。最终结论以校正后 p 值为准。
     - 报告内容：不一致对 (b, c)、准确率差（百分点）、精确/渐近/校正后 p 值。

数据来源：results 目录下 my_predict.py / my_predict.bat 生成的结果文件，
文件名形如 <原文件名>_<模型名>.xlsx，含 Unit、predict_answer、Correct_Answer 等列。
gpt-6.1-sol 已被 gpt-6.2-astra 替代，默认不参与检验。

用法：
    python my_statistical_tests.py                 # 默认排除 gpt-6.1-sol（已被 gpt-6.2-astra 替代）
    python my_statistical_tests.py --alpha 0.01      # 改显著性水平
    python my_statistical_tests.py --results-dir other_dir
    python my_statistical_tests.py --no-default-exclude  # 让 gpt-6.1-sol 也参与检验
    python my_statistical_tests.py --no-plot         # 只出表，不画柱状图

结果打印到终端，并保存为 results/stat_tests_cochran.csv 与 results/stat_tests_mcnemar.csv，
以及 Cochran's Q p 值柱状图 results/stat_tests_cochran.png（图中文字全部为英文）。
"""

import argparse
import glob
import os
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
COCHRAN_CSV = 'stat_tests_cochran.csv'
MCNEMAR_CSV = 'stat_tests_mcnemar.csv'
CHART_FILE = 'stat_tests_cochran.png'

UNIT_COLUMN = 'Unit'
STEM_COLUMN = 'Question_Stem'
PREDICT_COLUMN = 'predict_answer'
TRUTH_COLUMN = 'Correct_Answer'
ALL_UNITS_LABEL = '全部单元'

ALPHA = 0.05
N_PERMUTATIONS = 10000   # Cochran's Q 置换检验的置换次数
PERM_SEED = 42
# gpt-6.1-sol 已被 gpt-6.2-astra 替代，默认不参与统计检验
DEFAULT_EXCLUDED_MODELS = ('gpt-6.1-sol',)
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


# ---------------------------------------------------------------- 数据读取

def list_result_files(results_dir, excluded=()):
    """返回 [(模型名, 文件路径)]，模型名取自文件名最后一个下划线之后的部分。

    excluded 中的模型不参与检验（默认排除 gpt-6.1-sol）。
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


def load_correctness(results_dir, excluded=()):
    """读取所有模型的结果，返回 (Unit 列, {模型: 0/1 正确序列}, 题号列表)。

    各模型的结果文件必须逐行对齐（同一题在同一行），用 Question_Stem 校验。
    """
    files = list_result_files(results_dir, excluded)
    if len(files) < 2:
        raise FileNotFoundError(
            f'{results_dir} 下少于两个 <文件名>_<模型名>.xlsx 结果文件，无法做模型间比较')

    units = None
    question_nos = None
    reference_stems = None
    reference_name = None
    correctness = {}

    for model, path in files:
        df = pd.read_excel(path, usecols=[UNIT_COLUMN, 'Question_No', STEM_COLUMN,
                                          PREDICT_COLUMN, TRUTH_COLUMN])
        stems = df[STEM_COLUMN].astype(str).tolist()
        if reference_stems is None:
            reference_stems = stems
            reference_name = os.path.basename(path)
            units = df[UNIT_COLUMN].tolist()
            question_nos = df['Question_No'].tolist()
        elif stems != reference_stems:
            raise ValueError(
                f'{os.path.basename(path)} 的题目顺序与 {reference_name} 不一致，无法按题配对比较')

        predicted = df[PREDICT_COLUMN].astype(str).str.strip().str.upper()
        truth = df[TRUTH_COLUMN].astype(str).str.strip().str.upper()
        correctness[model] = (predicted == truth).astype(int).to_numpy()

    return units, question_nos, correctness


# ---------------------------------------------------------------- 检验方法

def cochran_q(matrix):
    """Cochran's Q 检验。matrix：n 道题 × k 个模型的 0/1 矩阵，返回 (Q, df, p)。"""
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    if n == 0 or k < 2:
        return float('nan'), max(k - 1, 0), float('nan')

    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    total = matrix.sum()

    denominator = k * total - np.sum(row_sums ** 2)
    if denominator <= 0:
        # 所有题目的模型间结果完全一致（或全对/全错），不存在差异
        return 0.0, k - 1, 1.0

    q = (k - 1) * (k * np.sum(col_sums ** 2) - total ** 2) / denominator
    return float(q), k - 1, float(chi2.sf(q, k - 1))


def permutation_q_test(matrix, n_permutations=N_PERMUTATIONS, seed=PERM_SEED):
    """Cochran's Q 的置换检验（Monte Carlo），作为大样本近似的稳健替代。

    零假设下每道题内部各模型的结果可交换：对每行随机重排 0/1（等价于把该题答对的
    次数随机分配给 k 个模型），重算 Q，得到 Q 的置换分布，再用 (1+超出次数)/(B+1) 估计 p。
    返回 (Q 观测值, 置换 p 值)。
    """
    matrix = np.asarray(matrix, dtype=int)
    n, k = matrix.shape
    q_observed, _, _ = cochran_q(matrix)
    if n == 0 or k < 2 or not np.isfinite(q_observed):
        return q_observed, float('nan')

    row_sums = matrix.sum(axis=1).astype(int)
    denominator = k * matrix.sum() - np.sum(row_sums ** 2)
    if denominator <= 0:
        return q_observed, 1.0

    rng = np.random.default_rng(seed)
    row_index = np.arange(n)[:, None]
    rank_base = np.arange(k)[None, :]
    exceed = 0
    for _ in range(n_permutations):
        # 每行随机排列 k 个位置，取前 row_sums 个为“答对”，等价于行内重排
        order = np.argsort(rng.random((n, k)), axis=1)
        ranks = np.empty_like(order)
        ranks[row_index, order] = rank_base
        col_sums = (ranks < row_sums[:, None]).sum(axis=0)
        total = col_sums.sum()
        q = (k - 1) * (k * np.sum(col_sums ** 2) - total ** 2) / denominator
        if q >= q_observed - 1e-12:
            exceed += 1

    p_value = (1 + exceed) / (n_permutations + 1)
    return q_observed, float(p_value)


def mcnemar_exact(correct_a, correct_b):
    """配对 McNemar 检验：返回 (b, c, 精确 p, 渐近 p)。

    b = 仅模型 A 答对的题数，c = 仅模型 B 答对的题数。
    精确 p 用二项检验（binomtest, p=0.5）；渐近 p 用带连续性校正的卡方。
    """
    a = np.asarray(correct_a, dtype=bool)
    b_array = np.asarray(correct_b, dtype=bool)

    only_a = int(np.sum(a & ~b_array))   # b
    only_b = int(np.sum(~a & b_array))   # c
    discordant = only_a + only_b

    if discordant == 0:
        return only_a, only_b, 1.0, 1.0

    p_exact = float(binomtest(only_a, discordant, 0.5).pvalue)
    chi2_stat = (abs(only_a - only_b) - 1) ** 2 / discordant
    p_asymptotic = float(chi2.sf(chi2_stat, 1))
    return only_a, only_b, p_exact, p_asymptotic


def holm_adjust(p_values):
    """Holm-Bonferroni 多重比较校正，返回与输入同序的校正后 p 值。"""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted





def format_p(p_value):
    """p 值显示：很小的时候写成 <0.0001。"""
    if not np.isfinite(p_value):
        return 'nan'
    if p_value < 0.0001:
        return '<0.0001'
    return f'{p_value:.4f}'


# ---------------------------------------------------------------- 图形输出

def plot_cochran_pvalues(cochran_rows, subtitle, alpha, output_file):
    """绘制 Cochran's Q 的 p 值柱状图（卡方近似 vs 置换检验），保存为 PNG。"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无界面环境也能保存图片
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f'绘制柱状图需要 matplotlib，当前环境不可用：{exc}') from exc

    plt.rcParams['font.family'] = 'DejaVu Sans'

    labels = ['All units' if r[0] == ALL_UNITS_LABEL else f'Unit {r[0]}' for r in cochran_rows]
    chi_p = [r[4] for r in cochran_rows]
    perm_p = [r[5] for r in cochran_rows]

    positions = np.arange(len(labels))
    bar_width = 0.35
    figure, axes = plt.subplots(figsize=(9.6, 7.0))

    chi_bars = axes.bar(positions - bar_width / 2, chi_p, width=bar_width,
                        color='#4C72B0', label='Chi-square approximation')
    perm_bars = axes.bar(positions + bar_width / 2, perm_p, width=bar_width,
                         color='#DD8452', label='Permutation test')

    for bars, values in ((chi_bars, chi_p), (perm_bars, perm_p)):
        for bar, value in zip(bars, values):
            value = min(value, 1.0)
            # 小 p 值的柱子很矮，文字抬到显著性水平线之上，避免与红色虚线重叠
            label_y = max(value, alpha) + 0.02
            axes.text(bar.get_x() + bar.get_width() / 2, label_y, f'{value:.3f}',
                      ha='center', va='bottom', fontsize=11, color='#222222', zorder=6)

    axes.axhline(alpha, color='red', linestyle='--', linewidth=1.2, zorder=5,
                 label=f'α = {alpha:g}')
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=12)
    axes.set_ylabel('p-value', fontsize=12)
    axes.set_ylim(0, 1.08)
    axes.set_yticks(np.arange(0, 1.01, 0.2))
    axes.tick_params(axis='y', labelsize=11)
    axes.yaxis.grid(True, linewidth=0.6, color='#DDDDDD')
    axes.set_axisbelow(True)
    axes.set_title("Cochran's Q test p-values (all units and per unit)\n" + subtitle,
                   fontsize=14, pad=10)
    axes.legend(loc='upper left', fontsize=11, framealpha=0.9)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close(figure)
    return output_file


# ---------------------------------------------------------------- 表格输出

def display_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1 for ch in str(text))


def print_table(headers, rows, aligns=None):
    """按显示宽度对齐打印表格（中文占两格）。"""
    if aligns is None:
        aligns = ['right'] * len(headers)
    widths = [max(display_width(headers[i]), *(display_width(r[i]) for r in rows)) if rows
              else display_width(headers[i]) for i in range(len(headers))]

    def render(cells):
        parts = []
        for cell, width, align in zip(cells, widths, aligns):
            cell = str(cell)
            gap = max(width - display_width(cell), 0)
            parts.append(cell + ' ' * gap if align == 'left' else ' ' * gap + cell)
        return '  '.join(parts).rstrip()

    print(render(headers))
    print('-' * max(display_width(render(headers)), 20))
    for row in rows:
        print(render(row))


# ---------------------------------------------------------------- 主流程

def main():
    parser = argparse.ArgumentParser(description='对各模型预测结果做 Cochran Q 与 McNemar 假设检验')
    parser.add_argument('--results-dir', default=RESULTS_DIR, help='结果文件目录')
    parser.add_argument('--output-dir', default=RESULTS_DIR, help='统计结果 CSV 输出目录')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='显著性水平')
    parser.add_argument('--n-permutations', type=int, default=N_PERMUTATIONS,
                        help='整体 Cochran Q 置换检验的置换次数')
    parser.add_argument('--seed', type=int, default=PERM_SEED, help='置换检验随机种子')
    parser.add_argument('--exclude', action='append', default=[],
                        help='不参与检验的模型名，可重复指定；默认已排除 gpt-6.1-sol')
    parser.add_argument('--no-default-exclude', action='store_true',
                        help='不使用默认排除列表（即让 gpt-6.1-sol 也参与检验）')
    parser.add_argument('--plot', default=None,
                        help='Cochran Q p 值柱状图输出路径（PNG），默认 <output-dir>/stat_tests_cochran.png')
    parser.add_argument('--no-plot', action='store_true', help='不绘制柱状图')
    args = parser.parse_args()

    excluded = [] if args.no_default_exclude else list(DEFAULT_EXCLUDED_MODELS)
    excluded += args.exclude or []

    units, question_nos, correctness = load_correctness(args.results_dir, excluded)
    raw_models = list(correctness.keys())
    models = [display_model_name(m) for m in raw_models]
    n = len(question_nos)
    matrix = np.column_stack([correctness[m] for m in raw_models])

    print(f'参与比较的模型（{len(models)} 个）：{", ".join(models)}')
    if excluded:
        print(f'已排除模型（不参与检验）：{", ".join(sorted(display_model_name(m) for m in set(excluded)))}')
    print(f'题目数：{n}（各模型逐题配对，题目顺序已校验一致）')
    print(f'显著性水平：{args.alpha}\n')

    # ---- 各模型准确率
    acc_rows = []
    for j, model in enumerate(models):
        hits = int(matrix[:, j].sum())
        acc_rows.append([model, n, hits, f'{hits / n * 100:.2f}%'])
    print('一、各模型准确率')
    print_table(['模型', '题数', '正确数', '准确率'], acc_rows, ['left', 'right', 'right', 'right'])
    print()

    # ---- Cochran's Q：整体与分单元（均附 Monte Carlo 置换检验 p 值）
    cochran_rows = []
    q_all, df_all, p_all = cochran_q(matrix)
    _, p_perm_all = permutation_q_test(matrix, args.n_permutations, args.seed)
    cochran_rows.append([ALL_UNITS_LABEL, n, q_all, df_all, p_all, p_perm_all])

    unit_order = []
    for unit in units:
        if unit not in unit_order:
            unit_order.append(unit)
    unit_order.sort(key=lambda u: (0, float(u)) if isinstance(u, (int, float)) else (1, str(u)))

    unit_index = np.array([str(u) for u in units])
    unit_sizes = []
    for unit in unit_order:
        mask = unit_index == str(unit)
        q_u, df_u, p_u = cochran_q(matrix[mask])
        _, p_perm_u = permutation_q_test(matrix[mask], args.n_permutations, args.seed)
        unit_sizes.append(int(mask.sum()))
        cochran_rows.append([str(unit), int(mask.sum()), q_u, df_u, p_u, p_perm_u])

    print(f"二、Cochran's Q 检验（多模型整体差异，H0：所有模型准确率相同；"
          f"置换检验为 Monte Carlo {args.n_permutations} 次；"
          f"{len(cochran_rows)} 个范围构成一个检验族，置换 p 值做 Holm-Bonferroni 校正，"
          f"结论以校正后 p 值为准）")
    # 全部单元与各单元共 len(cochran_rows) 次检验视为同一检验族，对置换 p 值做 Holm 校正
    adjusted = holm_adjust([r[5] for r in cochran_rows])
    for row, p_holm in zip(cochran_rows, adjusted):
        row.append(float(p_holm))
    print_table(['范围', '题数', 'Q 统计量', 'df', 'p 值(卡方)', 'p 值(置换)', 'p 值(Holm)', '结论'],
                [[r[0], r[1], f'{r[2]:.4f}', r[3], format_p(r[4]), format_p(r[5]),
                  format_p(r[6]),
                  '存在显著差异' if r[6] < args.alpha else '无显著差异']
                 for r in cochran_rows],
                ['left', 'right', 'right', 'right', 'right', 'right', 'right', 'left'])
    print()

    # ---- 两两 McNemar（全部单元 + 各单元），每个范围内分别 Holm 校正
    scopes = [(ALL_UNITS_LABEL, np.ones(n, dtype=bool))]
    for unit in unit_order:
        scopes.append((str(unit), unit_index == str(unit)))

    pair_rows = []
    for scope_label, scope_mask in scopes:
        scope_n = int(scope_mask.sum())
        scope_rows = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                only_i, only_j, p_exact, p_asymp = mcnemar_exact(
                    matrix[scope_mask, i], matrix[scope_mask, j])
                acc_diff = (only_i - only_j) / scope_n * 100
                scope_rows.append({
                    '范围': scope_label, '题数': scope_n,
                    '模型A': models[i], '模型B': models[j],
                    '仅A正确': only_i, '仅B正确': only_j,
                    '准确率差(pp)': acc_diff,
                    'p_精确': p_exact, 'p_渐近': p_asymp,
                })
        adjusted = holm_adjust([row['p_精确'] for row in scope_rows])
        for row, p_holm in zip(scope_rows, adjusted):
            row['p_Holm校正'] = float(p_holm)
        pair_rows.extend(scope_rows)

    all_scope_rows = [r for r in pair_rows if r['范围'] == ALL_UNITS_LABEL]
    print('三、两两 McNemar 检验（全部单元，Holm-Bonferroni 校正）')
    print_table(['模型 A', '模型 B', '仅A正确', '仅B正确', '准确率差(pp)', 'p 值(精确)', 'p 值(Holm)', '结论'],
                [[r['模型A'], r['模型B'], r['仅A正确'], r['仅B正确'],
                  f"{r['准确率差(pp)']:+.2f}", format_p(r['p_精确']),
                  format_p(r['p_Holm校正']),
                  '存在显著差异' if r['p_Holm校正'] < args.alpha else '无显著差异']
                 for r in all_scope_rows],
                ['left', 'left', 'right', 'right', 'right', 'right', 'right', 'left'])
    print()

    unit_scope_rows = [r for r in pair_rows if r['范围'] != ALL_UNITS_LABEL]
    print('四、两两 McNemar 检验（分单元，各单元内分别 Holm-Bonferroni 校正）')
    print_table(['范围', '模型 A', '模型 B', '仅A正确', '仅B正确', '准确率差(pp)', 'p 值(精确)', 'p 值(Holm)', '结论'],
                [[r['范围'], r['模型A'], r['模型B'], r['仅A正确'], r['仅B正确'],
                  f"{r['准确率差(pp)']:+.2f}", format_p(r['p_精确']),
                  format_p(r['p_Holm校正']),
                  '存在显著差异' if r['p_Holm校正'] < args.alpha else '无显著差异']
                 for r in unit_scope_rows],
                ['left', 'left', 'left', 'right', 'right', 'right', 'right', 'right', 'left'])
    print()

    # ---- 结论
    print('五、结论')
    print(f"  多模型整体检验：Cochran's Q = {q_all:.4f}, df = {df_all}, "
          f"p(卡方) = {format_p(p_all)}，p(置换) = {format_p(p_perm_all)}，"
          f"p(Holm，{len(cochran_rows)} 个范围) = {format_p(cochran_rows[0][6])}")
    if cochran_rows[0][6] < args.alpha:
        print(f'  → p(Holm) < {args.alpha}：所有模型准确率并非完全相同，整体上存在统计学显著差异。')
    else:
        print(f'  → p(Holm) >= {args.alpha}：对多个范围做 Holm 校正后，'
              f'未发现所有模型准确率存在整体统计学显著差异。')
    significant_pairs = [r for r in all_scope_rows if r['p_Holm校正'] < args.alpha]
    if significant_pairs:
        print('  事后两两比较·全部单元（Holm 校正后仍显著的模型对）：')
        for r in significant_pairs:
            better = r['模型A'] if r['准确率差(pp)'] > 0 else r['模型B']
            print(f"    {r['模型A']} vs {r['模型B']}：p(Holm) = {format_p(r['p_Holm校正'])}，"
                  f"准确率差 {r['准确率差(pp)']:+.2f} 个百分点（{better} 更高）")
    else:
        print('  事后两两比较·全部单元：所有模型对在 Holm 校正后均无显著差异'
              '（未校正的精确 p 值可作趋势参考，需更多数据才能确认差异）。')
    sig_unit_pairs = [r for r in unit_scope_rows if r['p_Holm校正'] < args.alpha]
    if sig_unit_pairs:
        print('  事后两两比较·分单元（单元内 Holm 校正后仍显著的模型对）：')
        for r in sig_unit_pairs:
            better = r['模型A'] if r['准确率差(pp)'] > 0 else r['模型B']
            print(f"    单元 {r['范围']}：{r['模型A']} vs {r['模型B']}："
                  f"p(Holm) = {format_p(r['p_Holm校正'])}，"
                  f"准确率差 {r['准确率差(pp)']:+.2f} 个百分点（{better} 更高）")
    else:
        print('  事后两两比较·分单元：各单元内所有模型对在 Holm 校正后均无显著差异。')

    # ---- 保存 CSV
    os.makedirs(args.output_dir, exist_ok=True)
    cochran_csv = os.path.join(args.output_dir, COCHRAN_CSV)
    mcnemar_csv = os.path.join(args.output_dir, MCNEMAR_CSV)

    pd.DataFrame({
        '范围': [r[0] for r in cochran_rows],
        '题数': [r[1] for r in cochran_rows],
        'Q统计量': [round(r[2], 6) for r in cochran_rows],
        'df': [r[3] for r in cochran_rows],
        'p值': [r[4] for r in cochran_rows],
        'p值_置换': [r[5] for r in cochran_rows],
        'p值_置换_Holm校正': [r[6] for r in cochran_rows],
        '显著': ['是' if r[6] < args.alpha else '否' for r in cochran_rows],
    }).to_csv(cochran_csv, index=False, encoding='utf-8-sig')

    pd.DataFrame([{
        '范围': r['范围'],
        '题数': r['题数'],
        '模型A': r['模型A'],
        '模型B': r['模型B'],
        '仅A正确': r['仅A正确'],
        '仅B正确': r['仅B正确'],
        '准确率差(pp)': round(r['准确率差(pp)'], 4),
        'p值_精确': r['p_精确'],
        'p值_渐近': r['p_渐近'],
        'p值_Holm校正': r['p_Holm校正'],
        '显著': '是' if r['p_Holm校正'] < args.alpha else '否',
    } for r in pair_rows]).to_csv(mcnemar_csv, index=False, encoding='utf-8-sig')

    print(f'\nCochran Q 结果已保存到 {cochran_csv}')
    print(f'McNemar 结果已保存到 {mcnemar_csv}')

    if not args.no_plot:
        per_unit = (str(unit_sizes[0]) if unit_sizes and len(set(unit_sizes)) == 1
                    else 'varying sizes')
        subtitle = f'{len(models)} models; {per_unit} per unit'
        chart_file = args.plot or os.path.join(args.output_dir, CHART_FILE)
        plot_cochran_pvalues(cochran_rows, subtitle, args.alpha, chart_file)
        print(f'柱状图已保存到 {chart_file}')


if __name__ == '__main__':
    main()
