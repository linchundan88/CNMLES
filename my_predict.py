# -*- coding: utf-8 -*-
"""对 Chinese_Dental_Licensing_Examination.xlsx 中的每道题目调用大语言模型进行预测。

模型：gpt-6.1-sol（temperature=0）
输出：results/<原文件名>_<模型名>.xlsx，
      即 results/Chinese_Dental_Licensing_Examination_gpt-6.1-sol.xlsx，
      在原始表格基础上新增一列 predict_answer（保留原表所有列与顺序）。

用法：
    python my_predict.py                # 全量预测
    python my_predict.py --limit 5      # 只预测前 5 题（调试用）
    python my_predict.py --model gemini-4-carbon --retry-failed
                                        # 只重新预测该模型已有结果中失败/无有效答案的题目，
                                        # 新结果写回原结果文件，其余题目保持不变
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from libs.my_helper_exam import concat_input_text, parse_result, system_instruction
from libs.my_helper_llm import get_llm_client, process_llm_prediction

QUESTION_STEM_FIELD = 'Question_Stem'
PREDICT_COLUMN = 'predict_answer'
VALID_ANSWERS = ('A', 'B', 'C', 'D', 'E')
MODEL_NAME = 'gpt-6.1-sol'
TEMPERATURE = 0
INPUT_FILE = os.path.join(BASE_DIR, 'datafiles', 'Chinese_Dental_Licensing_Examination.xlsx')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MAX_WORKERS = 8
CHECKPOINT_EVERY = 25
REQUEST_TIMEOUT = 120


def predict_one(client, model_name, temperature, record, index):
    """预测一道题，返回 (题号索引, 预测答案, 模型原始输出)。"""
    input_text = concat_input_text(record, QUESTION_STEM_FIELD)
    _, prediction = process_llm_prediction(
        client, model_name, system_instruction, input_text,
        temperature=temperature, index=index, max_retries=3,
    )

    if isinstance(prediction, str) and prediction.startswith('System Error'):
        # 调用失败时保留错误信息，便于后续排查
        predict_answer = prediction
    else:
        predict_answer = parse_result(prediction)
        if predict_answer is None:
            predict_answer = f'Error: empty prediction. {prediction}'

    return index, predict_answer, prediction


def save_result(df, predict_answers, output_file):
    """按原表列顺序输出，并在末尾追加 predict_answer 列。"""
    df_out = df.copy()
    df_out['predict_answer'] = [predict_answers.get(i, '') for i in range(len(df_out))]
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_excel(output_file, index=False)
    return df_out


def build_output_path(input_file, model_name):
    """结果文件名 = 原文件名 + '_' + 模型名，保存到 results 目录。"""
    stem, ext = os.path.splitext(os.path.basename(input_file))
    return os.path.join(RESULTS_DIR, f'{stem}_{model_name}{ext}')


def is_valid_answer(value):
    """预测值是否为合法选项字母（A-E），否则视为上次失败需要重跑。"""
    return isinstance(value, str) and value.strip().upper() in VALID_ANSWERS


def load_retry_targets(result_file):
    """读取已有结果文件，返回 (数据框, 待重新预测的行号列表)。

    待重跑的行 = predict_answer 不是 A-E 的行（包括空值、System Error、解析失败等）。
    """
    if not os.path.exists(result_file):
        raise FileNotFoundError(
            f'未找到已有结果文件：{result_file}\n'
            f'--retry-failed 需要先有一次预测结果；如需全量预测请去掉该参数。')

    df = pd.read_excel(result_file)
    if PREDICT_COLUMN not in df.columns:
        raise ValueError(f'结果文件缺少 {PREDICT_COLUMN} 列：{result_file}')

    pending = [i for i, value in enumerate(df[PREDICT_COLUMN].tolist()) if not is_valid_answer(value)]
    return df, pending


def main():
    parser = argparse.ArgumentParser(description='用大语言模型逐题预测口腔执业医师考试答案')
    parser.add_argument('--input', default=INPUT_FILE, help='输入 Excel 文件')
    parser.add_argument('--output', default='', help='输出 Excel 文件（默认 results/原文件名_模型名.xlsx）')
    parser.add_argument('--model', default=MODEL_NAME, help='模型名称')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='采样温度')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='并发线程数')
    parser.add_argument('--limit', type=int, default=0, help='只预测前 N 题（0 表示全部）')
    parser.add_argument('--retry-failed', action='store_true',
                        help='只重新预测已有结果文件中失败/无有效答案的题目，并把新结果写回该文件')
    args = parser.parse_args()

    output_file = args.output or build_output_path(args.input, args.model)
    start_time = time.time()

    if args.retry_failed:
        # 只重跑失败题：题目来自已有结果文件，其余行的预测结果原样保留
        df, pending_indices = load_retry_targets(output_file)
        print(f'已有结果文件：{output_file}（共 {len(df)} 行）')
        print(f'上次失败/无有效答案、需要重新预测的题目：{len(pending_indices)} 题')
        if not pending_indices:
            print('没有需要重跑的题目，结果文件保持不变。')
            return 0
        predict_answers = {i: df[PREDICT_COLUMN].iloc[i] for i in range(len(df))}
        for idx in pending_indices:
            predict_answers[idx] = ''
    else:
        df = pd.read_excel(args.input)
        if args.limit and args.limit > 0:
            df = df.head(args.limit).copy()
        pending_indices = list(range(len(df)))
        predict_answers = {}

    total = len(pending_indices)
    print(f'模型：{args.model}；temperature：{args.temperature}；并发：{args.workers}')
    print(f'输出文件：{output_file}')

    client = get_llm_client(args.model, timeout=REQUEST_TIMEOUT)
    finished = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(predict_one, client, args.model, args.temperature,
                        df.iloc[idx].fillna(''), idx): idx
            for idx in pending_indices
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, answer, _raw = future.result()
            except Exception as exc:  # 兜底，避免单题异常中断整体任务
                answer = f'System Error: {exc}'
                failed += 1
            else:
                if answer.startswith('System Error') or answer.startswith('Error:'):
                    failed += 1

            predict_answers[idx] = answer
            finished += 1
            flag = '' if not (answer.startswith('System Error') or answer.startswith('Error:')) else '  <失败>'
            print(f'[{finished}/{total}] 第 {idx + 1} 题 -> {answer}{flag}', flush=True)

            if finished % CHECKPOINT_EVERY == 0:
                save_result(df, predict_answers, output_file)
                print(f'  ...检查点已写入 {output_file}', flush=True)

    df_out = save_result(df, predict_answers, output_file)
    elapsed = time.time() - start_time
    action = '重新预测' if args.retry_failed else '预测'
    print(f'完成：本次{action} {total} 题，其中失败 {failed} 题，耗时 {elapsed:.1f} 秒')
    print(f'结果已保存到 {output_file}')

    if 'Correct_Answer' in df_out.columns:
        truth = df_out['Correct_Answer'].astype(str).str.strip().str.upper().tolist()
        row_count = len(df_out)
        valid_count = sum(1 for v in predict_answers.values() if is_valid_answer(v))
        hits = sum(1 for i, t in enumerate(truth)
                   if is_valid_answer(predict_answers.get(i))
                   and predict_answers[i].strip().upper() == t)
        print(f'预测准确率：{hits}/{row_count} = {hits / row_count * 100:.2f}%'
              f'（有效预测 {valid_count} 题）')


if __name__ == '__main__':
    main()
