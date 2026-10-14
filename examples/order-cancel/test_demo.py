"""端到端验证：训练、独立进程预测、HTTP 输入边界和重启一致性。"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
ORDER = {"amount": 500.0, "account_days": 30, "past_cancels": 2,
         "payment_channel": "wechat"}


def run(directory, filename):
    return subprocess.check_output(
        [sys.executable, filename], cwd=directory, text=True, timeout=120
    )


def post(url, body):
    request = Request(url + "/predict", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


@contextmanager
def server(directory):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
             "--port", str(port), "--no-access-log"],
            cwd=directory, stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    with urlopen(url + "/openapi.json", timeout=1):
                        yield url
                        return
                except (URLError, TimeoutError):
                    time.sleep(0.1)
            log.seek(0)
            raise RuntimeError("服务未正常启动：" + log.read()[-3000:])
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def check_prediction(result):
    assert math.isfinite(result["cancel_probability"])
    assert 0 <= result["cancel_probability"] <= 1
    assert result["predicted_cancel"] == (
        result["cancel_probability"] >= result["threshold"]
    )


def main():
    with tempfile.TemporaryDirectory(prefix="order-cancel-test-") as temp:
        directory = Path(temp)
        for name in ("train.py", "predict.py", "app.py", "orders.csv"):
            shutil.copy2(ROOT / name, directory / name)
        run(directory, "train.py")
        report = json.loads((directory / "evaluation.json").read_text())
        assert report["rows"] == {"train": 1800, "valid": 600, "test": 600}
        assert sum(map(sum, report["model"]["confusion_matrix"])) == 600
        first = json.loads(run(directory, "predict.py"))
        assert first == json.loads(run(directory, "predict.py"))
        check_prediction(first)

        with server(directory) as url:
            status, result = post(url, ORDER)
            assert status == 200 and result == first
            status, reordered = post(url, dict(reversed(list(ORDER.items()))))
            assert status == 200 and reordered == first
            for variant in ({**ORDER, "amount": None},
                            {**ORDER, "payment_channel": None},
                            {**ORDER, "payment_channel": "new_channel"}):
                status, result = post(url, variant)
                assert status == 200
                check_prediction(result)
            invalid = [
                {**ORDER, "amount": -1},
                {**ORDER, "amount": float("nan")},
                {**ORDER, "amount": float("inf")},
                {**ORDER, "account_days": "30"},
                {**ORDER, "past_cancels": 1.5},
                {**ORDER, "cancelled": 1},
                {**ORDER, "payment_channel": "   "},
                {name: None for name in ORDER},
                {name: value for name, value in ORDER.items() if name != "amount"},
            ]
            for variant in invalid:
                status, result = post(url, variant)
                assert status == 422 and "detail" in result, (variant, status, result)
        with server(directory) as url:
            status, result = post(url, ORDER)
            assert status == 200 and result == first
        print("PASS: 训练、独立加载、HTTP 输入校验、字段顺序和重启一致性")


if __name__ == "__main__":
    main()
