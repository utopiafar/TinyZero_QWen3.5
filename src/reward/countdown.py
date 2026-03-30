"""
Countdown 任务的 Reward 计算逻辑。

移植自 TinyZero 项目 (verl/utils/reward_score/countdown.py)

评分规则:
    - 正确等式 (数字正确 + 结果正确): 1.0
    - 格式正确但答案错误: 0.1
    - 无效格式: 0.0
"""

import re
import random


def extract_solution(solution_str):
    """从模型输出中提取 <answer> 标签内的等式。

    支持两种格式:
    1. "Assistant: ..." 格式 (base 模板)
    2. "<|im_start|>assistant ..." 格式 (ChatML 模板)
    """
    # 截取 assistant 回复部分
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    # 取最后一行（最终答案通常在最后）
    solution_str = solution_str.split('\n')[-1]

    # 提取 <answer>...</answer> 中的内容
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        return matches[-1].group(1).strip()
    return None


def validate_equation(equation_str, available_numbers):
    """验证等式只使用了给定的数字，且每个数字恰好使用一次。"""
    try:
        numbers_in_eq = sorted(int(n) for n in re.findall(r'\d+', equation_str))
        available_sorted = sorted(available_numbers)
        return numbers_in_eq == available_sorted
    except (ValueError, TypeError):
        return False


def evaluate_equation(equation_str):
    """安全地计算算术表达式的值。

    只允许数字、运算符 (+, -, *, /)、括号和空格。
    """
    try:
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.0):
    """计算 Countdown 任务的 reward 分数。

    Args:
        solution_str: 模型生成的完整回复文本
        ground_truth: dict, 包含:
            - target: 目标数值
            - numbers: 可用数字列表
        method: 提取方法 (当前仅支持 'strict')
        format_score: 格式正确但答案错误的分数
        score: 完全正确的分数

    Returns:
        float: reward 分数 (0.0, 0.1, 或 1.0)
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']

    # 随机打印样本用于调试 (约 1/64 概率)
    do_print = random.randint(1, 64) == 1

    equation = extract_solution(solution_str=solution_str)

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print("No equation found")
        return 0.0

    # 验证等式使用了正确的数字
    if not validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation (wrong numbers)")
        return format_score

    # 计算等式结果
    result = evaluate_equation(equation)
    if result is None:
        if do_print:
            print("Could not evaluate equation")
        return format_score

    # 比较结果（允许浮点误差）
    if abs(result - target) < 1e-5:
        if do_print:
            print(f"Correct! {equation} = {result}")
        return score
    else:
        if do_print:
            print(f"Wrong result: {equation} = {result}, target = {target}")
        return format_score
