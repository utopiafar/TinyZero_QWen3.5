"""
Qwen3-1.7B-Base Tokenizer 工具。

处理 Qwen3 的特殊 token:
- eos_token_id: 151643 (``)
- pad_token_id: 151643 (设为 eos)
- <|im_start|>: 151644
- <|im_end|>: 151645
"""


# Qwen3-1.7B-Base 特殊 token ID
# 来源: Qwen3-1.7B-Base/tokenizer_config.json
QWEN3_EOS_TOKEN_ID = 151643
QWEN3_PAD_TOKEN_ID = 151643
QWEN3_IM_START_ID = 151644
QWEN3_IM_END_ID = 151645

# Stop token IDs: EOS + <|im_end|>
# <|im_end|> 作为 stop token 可以防止模型继续生成下一个 turn
QWEN3_STOP_TOKEN_IDS = [QWEN3_EOS_TOKEN_ID, QWEN3_IM_END_ID]


def get_special_token_ids():
    """返回 Qwen3-1.7B-Base 的特殊 token ID 字典。"""
    return {
        "eos_token_id": QWEN3_EOS_TOKEN_ID,
        "pad_token_id": QWEN3_PAD_TOKEN_ID,
        "im_start_id": QWEN3_IM_START_ID,
        "im_end_id": QWEN3_IM_END_ID,
        "stop_token_ids": QWEN3_STOP_TOKEN_IDS,
    }


def setup_tokenizer(tokenizer):
    """配置 tokenizer 以适配 Qwen3-1.7B-Base。

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
