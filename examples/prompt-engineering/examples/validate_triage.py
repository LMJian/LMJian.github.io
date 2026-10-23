from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Triage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    category: Literal["upstream_timeout", "db_pool_exhausted", "unknown"]
    evidence_ids: list[str] = Field(max_length=3)
    missing_info: list[str] = Field(max_length=3)
    next_step: str = Field(min_length=1, max_length=100)


def validate_response(raw: str, logs: dict[str, str]) -> Triage:
    result = Triage.model_validate_json(raw)
    if not set(result.evidence_ids).issubset(logs):
        raise ValueError("响应引用了输入中不存在的日志编号")
    if result.category != "unknown" and not result.evidence_ids:
        raise ValueError("给出明确分类时，必须引用日志证据")
    return result


if __name__ == "__main__":
    logs = {"L1": "call inventory-api failed: read timeout after 200ms"}
    raw = '''{"category":"upstream_timeout","evidence_ids":["L1"],
              "missing_info":["上游同期延迟指标"],
              "next_step":"查询 inventory-api 同期的延迟和错误率"}'''
    result = validate_response(raw, logs)
    print(result.model_dump_json())
