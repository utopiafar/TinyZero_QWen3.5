"""
Countdown 任务的 Prompt 模板。

专为 Qwen3.5-2B-Base 设计:
- 使用 ChatML 格式 (<|im_start|> / <|im_end|>)
- 中文界面
- 包含 few-shot 示例
- 使用 <think reasoning> 标记 (避免触发 Qwen3.5 特殊 token)
"""

# Qwen3.5-2B-Base 的 ChatML 特殊 token
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Few-shot 示例数据
FEW_SHOT_EXAMPLE = {
    "numbers": [3, 7, 5, 2],
    "target": 24,
    "reasoning": (
        "3 * 7 = 21，然后 5 - 2 = 3，21 + 3 = 24。\n"
        "验证：3 * 7 + 5 - 2 = 21 + 5 - 2 = 24。正确！"
    ),
    "answer": "3 * 7 + 5 - 2",
}


def make_prefix(dp):
    """根据数据样本生成 ChatML 格式的对话提示。

    Args:
        dp: dict, 包含:
            - 'target': 目标数值
            - 'nums': 可用数字列表

    Returns:
        str: 格式化的对话提示字符串

    输出结构:
        system → 用户任务描述
        few-shot 用户问题 → few-shot 助手回答
        实际用户问题 → 预填充 "<think reasoning>\\n"
    """
    target = dp['target']
    numbers = dp['nums']

    prompt = (
        f"{IM_START}system\n"
        f"你是一个有帮助的助手。你会先在脑海中思考推理过程，然后给出答案。{IM_END}\n"

        # Few-shot 示例
        f"{IM_START}user\n"
        f"使用数字 {FEW_SHOT_EXAMPLE['numbers']}，通过基本算术运算（+、-、*、/）"
        f"构造一个等于 {FEW_SHOT_EXAMPLE['target']} 的等式。"
        f"每个数字只能使用一次。"
        f"请在 <think reasoning> </think reasoning> 标签中展示你的推理过程，"
        f"并在 <answer> </answer> 标签中返回最终答案，"
        f"例如 <answer> (1 + 2) / 3 </answer>。{IM_END}\n"

        f"{IM_START}assistant\n"
        f"<think reasoning>\n"
        f"{FEW_SHOT_EXAMPLE['reasoning']}\n"
        f"</think reasoning>\n"
        f"<answer> {FEW_SHOT_EXAMPLE['answer']} </answer>{IM_END}\n"

        # 实际问题
        f"{IM_START}user\n"
        f"使用数字 {numbers}，通过基本算术运算（+、-、*、/）"
        f"构造一个等于 {target} 的等式。"
        f"每个数字只能使用一次。"
        f"请在 <think reasoning> </think reasoning> 标签中展示你的推理过程，"
        f"并在 <answer> </answer> 标签中返回最终答案，"
        f"例如 <answer> (1 + 2) / 3 </answer>。{IM_END}\n"

        f"{IM_START}assistant\n"
        f"<think reasoning>\n"
    )

    return prompt
