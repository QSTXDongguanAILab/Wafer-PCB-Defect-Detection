"""一键启动:选解释器 → 查依赖 → 挑端口 → 起服务 → 开浏览器。

给双击启动用(见仓库根目录 `启动.bat`),也可以直接跑:
    python scripts/launch.py
    python scripts/launch.py --reload --no-browser --port 8790

为什么不写在 .bat 里:项目路径含中文,cmd 的代码页很容易把中文路径搞坏。
.bat 只做「找到一个能用的 python」这一件 ASCII 能表达的事,其余判断都在这里。
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 起 Web 服务的最小依赖。torch / ultralytics 只有训练和推理才需要,
# 缺了不该挡住看板启动 —— 权重本来就可能还没训。
WEB_DEPS = ("fastapi", "uvicorn", "yaml", "numpy", "cv2")
RELAUNCH_FLAG = "WPDD_RELAUNCHED"

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def missing_deps(python: str | None = None) -> list[str]:
    """列出缺失的 Web 依赖。python=None 表示检查当前解释器。"""
    if python is None:
        import importlib.util

        return [m for m in WEB_DEPS if importlib.util.find_spec(m) is None]
    code = (
        "import importlib.util,sys;"
        f"mods={list(WEB_DEPS)!r};"
        "print(','.join(m for m in mods if importlib.util.find_spec(m) is None))"
    )
    try:
        out = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return list(WEB_DEPS)
    if out.returncode != 0:
        return list(WEB_DEPS)
    return [m for m in out.stdout.strip().split(",") if m]


def candidate_interpreters() -> list[Path]:
    """按优先级找解释器。

    本仓库还没单独建 venv,所以第二顺位是旁边 ShopInspect 的 venv
    （torch / ultralytics / langchain 都装在那里）。路径在 Python 里拼,
    不经过 cmd 的代码页,中文目录名不会出问题。
    """
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return [
        ROOT / ".venv" / sub,
        ROOT.parent / "AI视觉" / "ShopInspect" / ".venv" / sub,
    ]


def relaunch_if_needed() -> None:
    """当前解释器缺依赖时,换一个装好的重新执行本脚本。

    用 subprocess 而不是 os.execv:Windows 上 exec 系列不真的替换进程映像,
    而是另起一个新 PID 后让当前进程立刻退出 —— 结果 .bat 会误以为服务已经停了,
    打出 "Server stopped" 并等按键,而服务其实还在后台跑,窗口里的 Ctrl+C 也管不到它。
    """
    if not missing_deps() or os.environ.get(RELAUNCH_FLAG):
        return
    for python in candidate_interpreters():
        if not python.exists() or missing_deps(str(python)):
            continue
        log(f"当前解释器缺依赖,改用: {python}")
        env = {**os.environ, RELAUNCH_FLAG: "1"}
        proc = subprocess.Popen(
            [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env
        )
        # Ctrl+C 由控制台送给整个进程组,子进程自己会收到;
        # 这里只负责继续等它退干净,最多容忍三次打断。
        for _ in range(3):
            try:
                raise SystemExit(proc.wait())
            except KeyboardInterrupt:
                continue
        proc.kill()
        raise SystemExit(130)
    # 一个都没找到,把话说清楚再退出,不要抛一堆 ImportError
    log("启动失败:没找到装好依赖的 Python 环境。")
    log(f"  当前解释器: {sys.executable}")
    log(f"  缺少模块:   {', '.join(missing_deps())}")
    log("")
    log("在项目目录下执行:")
    log("  python -m venv .venv")
    log(r"  .venv\Scripts\activate")
    log("  pip install -r requirements.txt")
    raise SystemExit(2)


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.6)
        return s.connect_ex((host or "127.0.0.1", port)) != 0


def pick_port(host: str, want: int) -> int:
    """配置端口被占就顺延,避免双击后报 10048 又不知道为什么。"""
    for port in range(want, want + 21):
        if port_free(host, port):
            if port != want:
                log(f"端口 {want} 已被占用,改用 {port}")
            return port
    raise SystemExit(f"启动失败:{want}~{want + 20} 全被占用,先关掉占用端口的进程。")


def open_browser_when_up(url: str, health: str, timeout: float = 40.0) -> None:
    """等 /health 通了再开浏览器,不然会开出一个连接失败的页面。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=1.5) as r:
                if r.status == 200:
                    webbrowser.open(url)
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.5)
    log(f"服务在 {timeout:.0f} 秒内没起来,浏览器就不自动开了,看上面的报错。")


def main() -> None:
    relaunch_if_needed()
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    from app.config import load_settings

    s = load_settings()
    ap = argparse.ArgumentParser(description="一键启动 Wafer-PCB-Defect-Detection")
    ap.add_argument("--host", default=s.host)
    ap.add_argument("--port", type=int, default=s.port)
    ap.add_argument("--reload", action="store_true", help="改代码自动重载(开发用)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    port = pick_port(args.host, args.port)
    open_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{open_host}:{port}/"

    from pcb.infer import get_predictor
    from wafer.infer import get_detector

    pcb_ready = get_predictor().ready
    wafer_ready = get_detector().ready
    try:
        from pcb.dataset import collect_pairs, describe

        snap = describe(collect_pairs(s.resolve(s.pcb.dataset_root)))
        data_line = f"PCB 数据:已标注 {snap['labeled']} 对 / 待标注 {snap['unlabeled']} 对"
    except Exception as exc:  # noqa: BLE001  数据盘没挂上也要能起服务
        data_line = f"PCB 数据:读不到（{type(exc).__name__}）"

    log("=" * 58)
    log(f"  {s.app_name}  v{__import__('app').__version__}")
    log("=" * 58)
    log(f"  解释器    {sys.executable}")
    log(f"  PCB 模型  {'已就绪' if pcb_ready else '未训练 → /pcb/inspect 会返回 503'}")
    log(f"  硅片模型  {'已就绪' if wafer_ready else '未训练 → /wafer/inspect 会返回 503'}")
    log(f"  {data_line}")
    log(f"  看板      {url}")
    log("=" * 58)
    if args.host not in ("127.0.0.1", "localhost"):
        log("⚠ 监听地址不是 127.0.0.1,局域网内其他机器可直接访问。")
        log("  本服务没有登录与鉴权,任何人都能上传图片、读写检测记录、删除数据。")
        log("  要对外提供请先加认证或放到反向代理后面。")
        log("=" * 58)
    log("按 Ctrl+C 停止服务。")
    log("")

    if not args.no_browser:
        threading.Thread(
            target=open_browser_when_up, args=(url, f"http://{open_host}:{port}/health"), daemon=True
        ).start()

    import uvicorn

    try:
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=port,
            reload=args.reload,
            reload_dirs=[str(ROOT / "app"), str(ROOT / "pcb"), str(ROOT / "wafer")]
            if args.reload
            else None,
            log_level="info",
        )
    except KeyboardInterrupt:
        log("\n已停止。")


if __name__ == "__main__":
    main()
