"""生成可重复的模拟订单；不包含真实用户或真实业务效果。"""
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(2026)
    n = 3000
    amount = rng.uniform(20, 1000, n).round(2)
    days = rng.integers(1, 366, n)
    cancels = rng.poisson(1, n)
    channel = rng.choice(["alipay", "wechat", "bank_card"], n)
    # 教学用生成规律，不是对任何支付渠道或用户群体的业务结论。
    z = -2 + 0.001 * amount - 0.006 * days + 0.9 * cancels
    z += np.where(channel == "bank_card", 0.3, 0.0)
    label = rng.binomial(1, np.exp(-np.logaddexp(0.0, -z)))
    frame = pd.DataFrame({
        "order_id": [f"demo-{i:04d}" for i in range(n)],
        "amount": amount, "account_days": days, "past_cancels": cancels,
        "payment_channel": channel, "cancelled": label,
    })
    # 模拟特征缺失；标签保持完整。缺失发生机制在这里被刻意简化。
    for name in ("amount", "account_days", "payment_channel"):
        frame[name] = frame[name].astype(object)
        frame.loc[rng.random(n) < 0.04, name] = None
    path = Path(__file__).with_name("orders.csv")
    frame.to_csv(path, index=False)
    print(f"写入 {len(frame)} 条模拟订单：{path.name}")


if __name__ == "__main__":
    main()
