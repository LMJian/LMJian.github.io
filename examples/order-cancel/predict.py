"""训练和 HTTP 服务共用输入约定；也可以作为独立预测程序运行。"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

NUMERIC = ["amount", "account_days", "past_cancels"]
CATEGORICAL = ["payment_channel"]
FEATURES = NUMERIC + CATEGORICAL
MODEL_PATH = Path(__file__).with_name("model.joblib")


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    # 字段必须提供；null 表示值暂缺，交给训练好的填补器处理。
    amount: float | None = Field(ge=0, allow_inf_nan=False)
    account_days: int | None = Field(ge=0)
    past_cancels: int | None = Field(ge=0)
    payment_channel: str | None = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_one_feature(self):
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("至少需要一个非空特征")
        return self


def to_frame(orders):
    rows = [
        {key: np.nan if value is None else value
         for key, value in order.model_dump().items()}
        for order in orders
    ]
    return pd.DataFrame(rows, columns=FEATURES)


class Predictor:
    def __init__(self, path=MODEL_PATH):
        # 只加载自己训练或已确认可信的 joblib 文件。
        bundle = joblib.load(path)
        if bundle["features"] != FEATURES:
            raise ValueError("模型特征约定与服务不一致")
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.version = bundle["model_version"]
        self.positive_index = list(self.pipeline.classes_).index(1)

    def predict(self, order):
        probability = float(self.pipeline.predict_proba(to_frame([order]))[
            0, self.positive_index
        ])
        return {
            "cancel_probability": probability,
            "threshold": self.threshold,
            "predicted_cancel": probability >= self.threshold,
            "model_version": self.version,
        }


if __name__ == "__main__":
    order = Order(amount=500.0, account_days=30, past_cancels=2,
                  payment_channel="wechat")
    print(json.dumps(Predictor().predict(order), ensure_ascii=False, indent=2))
