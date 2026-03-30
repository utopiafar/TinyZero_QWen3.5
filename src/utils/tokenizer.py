"""
Qwen3.5-2B-Base Tokenizer 工具。

处理 Qwen3.5 的特殊 token:
- eos_token_id: 248044 (``)
- pad_token_id: 248044 (设为 eos)
- <|im_start|>: 248045
- <|im_end|>: 248046
"""


# Qwen3.5-2B-Base 特殊 token ID
# 来源: https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/main/tokenizer_config.json
QWEN35_EOS_TOKEN_ID = 248044
QWEN35_PAD_TOKEN_ID = 248044
QWEN35_IM_START_ID = 248045
QWEN35_IM_END_ID = 248046

# Stop token IDs: EOS + <|im_end|>
# <|im_end|> 作为 stop token 可以防止模型继续生成下一个 turn
QWEN35_STOP_TOKEN_IDS = [QWEN35_EOS_TOKEN_ID, QWEN35_IM_END_ID]


def get_special_token_ids():
    """返回 Qwen3.5-2B-Base 的特殊 token ID 字典。"""
    return {
        "eos_token_id": QWEN35_EOS_TOKEN_ID,
        "pad_token_id": QWEN35_PAD_TOKEN_ID,
        "im_start_id": QWEN35_IM_START_ID,
        "im_end_id": QWEN35_IM_END_ID,
        "stop_token_ids": QWEN35_STOP_TOKEN_IDS,
    }


def setup_tokenizer(tokenizer):
    """配置 tokenizer 以适配 Qwen3.5-2B-Base。

    Args:
        tokenizer: HuggingFace PreTrainedTokenizer 实例

    Returns:
        tokenizer: 配置后的 tokenizer
    """
    # 确保 pad_token 设置正确
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
