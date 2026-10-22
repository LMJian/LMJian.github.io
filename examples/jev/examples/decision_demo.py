"""默认使用构造数据；也可读取已保存的 API 响应。本程序不发起网络请求。"""
import argparse
import json
import math
from pathlib import Path


def softmax(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    scaled = [z / temperature for z in logits]
    peak = max(scaled)
    weights = [math.exp(z - peak) for z in scaled]
    total = sum(weights)
    return [w / total for w in weights]


def route(answer):
    selected = answer["choice"]
    p_selected = answer["probabilities"][selected]
    if selected == "other" or p_selected < 0.70:
        return "human_review"
    if p_selected < 0.90:
        return "collect_more_context"
    return {"billing": "billing_queue", "technical": "technical_queue"}[selected]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, help="读取 call_jev.sh 保存的 response.json")
    args = parser.parse_args()
    if args.response:
        response = json.loads(args.response.read_text(encoding="utf-8"))
        answer = response["answers"]["team"]
    else:
        answer = {
            "choice": "billing",
            "probabilities": {"billing": 0.92, "technical": 0.05, "other": 0.03},
            "confidence": 0.88,
        }
        print("候选概率：", [round(p, 3) for p in softmax([3.0, 1.0, 0.0])])
    print("工单去向：", route(answer))
