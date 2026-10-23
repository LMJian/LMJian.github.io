"""Python 3.10+。仅生成本地草稿；用持久化状态演示恢复与幂等键。"""
import json
import sys
from pathlib import Path

state_file = Path(sys.argv[1] if len(sys.argv) > 1 else "task-state.json")
task_id = "daily-brief:2026-10-23"
state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
if state.get("task_id") == task_id:
    print(f"恢复已有任务：{state['status']}，未重复生成草稿")
else:
    # 真实系统在这里调用有授权的只读工具，并记录每条信息的来源。
    facts = ["订单服务错误率恢复到 0.1%", "支付变更等待评审"]
    draft = "今日摘要：\n" + "\n".join(f"- {fact}" for fact in facts)
    state = {
        "task_id": task_id,
        "status": "awaiting_approval",
        "draft": draft,
        "next_action": "发送前展示收件人和正文，等待人工批准",
    }
    temporary = state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(state_file)
    print(draft)
    print(state["next_action"])
