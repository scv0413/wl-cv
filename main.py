"""프로젝트 루트에서 fastapi dev로 실행하는 로컬 개발 진입점."""

import os
import sys
import time
import webbrowser
from multiprocessing import current_process
from pathlib import Path
from socket import create_connection
from threading import Thread
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv

# 원판 모델과 결과 저장 경로를 API가 처음 읽기 전에 설정한다.
cli_port = os.environ.get("PORT", "8000")
load_dotenv(Path(__file__).with_name(".env"))
from wlcv.api.app import app


def open_swagger_when_ready(port: int) -> None:
    url = f"http://127.0.0.1:{port}/docs"
    for _ in range(120):
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    break
        except (URLError, OSError):
            pass
        time.sleep(0.5)
    else:
        return
    webbrowser.open_new_tab(url)


# CLI가 앱을 처음 찾는 프로세스에서만 열어, 코드가 갱신돼도 탭을 추가하지 않는다.
if (
    sys.argv[1:] == ["dev"]
    and Path(sys.argv[0]).stem == "fastapi"
    and current_process().name == "MainProcess"
):
    port = int(cli_port)
    try:
        with create_connection(("127.0.0.1", port), timeout=0.3):
            pass  # 이미 사용 중인 포트의 문서를 잘못 열지 않는다.
    except ConnectionRefusedError:
        Thread(target=open_swagger_when_ready, args=(port,), daemon=True).start()
