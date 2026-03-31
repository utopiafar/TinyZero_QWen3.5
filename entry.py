"""
TinyZero QWen3 GRPO 训练入口。

在调用 verl 主流程前，注入自定义指标（格式成功率、结果成功率等）
到 TensorBoard 日志系统。
"""

import numpy as np


def _patch_compute_data_metrics():
    """Monkey-patch verl 的 compute_data_metrics，额外记录 reward 细分指标。

    verl 原生的 compute_data_metrics 只记录 score/reward/advantage 的统计量。
    我们在它的基础上，从 batch.non_tensor_batch 中提取 reward_extra_info
    （由 NaiveRewardManager 从 compute_score 返回的 dict 中收集），
    计算格式成功率和结果成功率，注入到 metrics 字典中。
    """
    import verl.trainer.ppo.metric_utils as _mu

    _original = _mu.compute_data_metrics

    def _patched(batch, use_critic=True):
        metrics = _original(batch, use_critic=use_critic)

        ntb = getattr(batch, "non_tensor_batch", {})
        for key in ("format_success", "result_success"):
            if key in ntb:
                vals = np.array(ntb[key], dtype=np.float64)
                if len(vals) > 0:
                    metrics[f"reward/{key}_rate"] = float(vals.mean())

        return metrics

    _mu.compute_data_metrics = _patched
    print("[entry.py] Patched compute_data_metrics with reward extra info logging")


def main():
    _patch_compute_data_metrics()

    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
