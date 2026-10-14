"""本机教学服务：启动时加载模型，每次请求只执行推理。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from predict import Order, Predictor


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor()
    yield


app = FastAPI(title="订单取消预测示例", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # 不回显原始输入，非有限数值等非法输入也能返回合法 JSON。
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict")
def predict(order: Order, request: Request):
    return request.app.state.predictor.predict(order)
