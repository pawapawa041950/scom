"""Embedded ComfyUI backend: launch as a local subprocess and drive via API.

Responsibilities:
  * Generate an ``extra_model_paths.yaml`` mapping our models/{diffusion,vae,te}
    folders onto ComfyUI's expected categories.
  * Start ``main.py`` as a subprocess bound to 127.0.0.1 on a chosen port.
  * Wait until the HTTP server is reachable.
  * Queue prompts, stream progress over the websocket, and return the decoded
    output image bytes.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import websocket  # websocket-client

from . import config
from .comfy_custom_nodes import ensure_custom_nodes
from .textutil import strip_ansi


def _free_port(preferred: int = 8199) -> int:
    """Return a usable localhost port, preferring ``preferred``."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("could not allocate a localhost port")


# ----- 残留バックエンド対策 ---------------------------------------------------
# scom が異常終了（強制終了・クラッシュ・シャットダウン）すると ComfyUI の
# 子プロセスが取り残され、モデルを RAM/VRAM に抱えたまま、ポートと
# ComfyUI の SQLite DB（comfyui.db.lock）を掴み続ける。次回起動は別ポートに
# 逃げるので動きはするが、「Database is locked」が出て遅くなる。
#   1. Windows のジョブオブジェクト（KILL_ON_JOB_CLOSE）で scom が死ねば
#      バックエンドの木ごと OS に消させる。
#   2. 起動時に前回の PID ファイルを見て、持ち主の scom が居なければ掃除する
#      （1. が効かなかった場合や旧版の取り残し用）。
#   3. ComfyUI の DB はメモリ上にする（scom はアセット DB を使わない）ので、
#      多重起動しても DB ロックで待たされることが無い。
_PID_FILE = "comfyui.pid.json"
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


def _pid_alive(pid: int) -> bool:
    """Windows: その PID のプロセスがまだ動いているか（終了済み/不在は False）。"""
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def _listener_pid(port: int) -> Optional[int]:
    """127.0.0.1:port で LISTEN しているプロセスの PID（netstat 経由）。"""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    needle = f":{port}"
    for line in out.splitlines():
        cols = line.split()
        if (len(cols) >= 5 and cols[0].upper() == "TCP"
                and cols[1].endswith(needle) and cols[3] == "LISTENING"):
            try:
                return int(cols[4])
            except ValueError:
                return None
    return None


def _attach_kill_on_close_job(proc: subprocess.Popen):
    """子プロセスをジョブに入れ、ジョブハンドルが閉じたら（= scom が
    どんな死に方をしても）木ごと強制終了させる。戻り値はハンドル（保持
    し続けること）。失敗時は None（既存ジョブの制約下など）。"""
    if sys.platform != "win32" or os.environ.get("SCOM_NO_JOB"):
        return None
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                            ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
    ok = (k32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                      ctypes.sizeof(info))   # ExtendedLimitInformation
          and k32.AssignProcessToJobObject(job, wintypes.HANDLE(proc._handle)))
    if not ok:
        k32.CloseHandle(job)
        return None
    return job


def write_extra_model_paths(paths: config.AppPaths) -> Path:
    """Write extra_model_paths.yaml pointing ComfyUI at our models folder.

    Also installs the scom custom nodes (ScomMergeModel) and registers their
    directory — an absolute path passes through ComfyUI's os.path.join with
    base_path unchanged, so it can live outside the models tree.
    """
    models = paths.models
    nodes_dir = ensure_custom_nodes(paths)
    yaml_text = (
        "scom:\n"
        f"  base_path: {models.as_posix()}\n"
        "  is_default: true\n"
        "  diffusion_models: diffusion_models/\n"
        # フルチェックポイント（VAE/CLIP 内蔵）を CheckpointLoaderSimple から
        # 読めるよう、同じ diffusion_models フォルダを checkpoints にも割り当てる。
        "  checkpoints: diffusion_models/\n"
        "  vae: vae/\n"
        # プロンプト整形用 LLM (models/llm) も CLIPLoader から読めるように
        # text_encoders の追加パスにする（ComfyUI は改行区切りで複数パス可）。
        "  text_encoders: |\n"
        "    text_encoders/\n"
        "    llm/\n"
        "  loras: loras/\n"
        f"  custom_nodes: {nodes_dir.as_posix()}\n"
    )
    out = paths.user_data / "extra_model_paths.yaml"
    out.write_text(yaml_text, encoding="utf-8")
    return out


@dataclass
class Progress:
    """A progress update streamed from the backend during generation."""
    value: int = 0
    maximum: int = 0
    note: str = ""


# 生成直後の一時的な接続断（WinError 10054 / タイムアウト）に対する GET の
# 再試行設定。ComfyUI 本体は生きているのに一瞬だけ応答できなくなるため。
_HTTP_ATTEMPTS = 5
_HTTP_BACKOFF = 0.4  # seconds; doubles each attempt (0.4, 0.8, 1.6, 3.2, 4.0)
_RETRY_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, OSError,
                 http.client.HTTPException)


class BackendError(RuntimeError):
    pass


class ComfyBackend:
    """Manages the ComfyUI subprocess and exposes a simple generate() API."""

    def __init__(self, paths: Optional[config.AppPaths] = None, port: int = 8199):
        self.paths = paths or config.AppPaths()
        self.port = port
        self.host = "127.0.0.1"
        self.client_id = uuid.uuid4().hex
        # 起動前にメインウィンドウが設定から反映する（次回起動時に有効）。
        self.use_sage_attention = False
        self.use_ck_attention = False
        self._proc: Optional[subprocess.Popen] = None
        self._job = None   # Windows ジョブオブジェクト（_attach_kill_on_close_job）
        self._log_thread: Optional[threading.Thread] = None
        self._log_tail: deque[str] = deque(maxlen=40)

    # ----- lifecycle -------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ----- 残留バックエンドの掃除 ------------------------------------------
    @property
    def _pid_file(self) -> Path:
        return self.paths.backend_root / _PID_FILE

    def _write_pid_file(self) -> None:
        try:
            self._pid_file.write_text(json.dumps({
                "owner_pid": os.getpid(), "pid": self._proc.pid,
                "port": self.port}), encoding="utf-8")
        except OSError:
            pass

    def _reap_stale_backend(self, log: Callable[[str], None]) -> None:
        """前回の scom が後始末できずに残した ComfyUI を終了させる。

        PID ファイルの持ち主（scom）がまだ生きていれば別インスタンスなので
        触らない。死んでいれば、記録したポートで応答している ComfyUI と
        記録した PID の木を止める。"""
        try:
            info = json.loads(self._pid_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        try:
            owner, pid, port = (int(info.get("owner_pid", 0)),
                                int(info.get("pid", 0)), int(info.get("port", 0)))
        except (TypeError, ValueError):
            self._pid_file.unlink(missing_ok=True)
            return
        if owner == os.getpid() or (owner and _pid_alive(owner)):
            return  # 自分、または起動中の別インスタンスのもの
        victims = []
        if port:
            try:
                with urllib.request.urlopen(
                        f"http://{self.host}:{port}/system_stats", timeout=2):
                    lp = _listener_pid(port)
                    if lp:
                        victims.append(lp)
            except (urllib.error.URLError, OSError):
                pass
        if pid and _pid_alive(pid) and pid not in victims:
            victims.append(pid)
        for v in victims:
            _kill_tree(v)
        if victims:
            log(f"前回の残留 ComfyUI プロセスを終了しました (PID {victims})")
            time.sleep(0.5)  # ポート解放を待つ
        self._pid_file.unlink(missing_ok=True)

    def start(self, log: Optional[Callable[[str], None]] = None,
              timeout: float = 120.0) -> None:
        """Launch ComfyUI and block until it responds, or raise BackendError."""
        log = log or (lambda _m: None)
        if self.is_running():
            return

        comfy = self.paths.comfyui
        if comfy is None:
            raise BackendError(
                "ComfyUI が見つかりません。SCOM_COMFYUI_DIR を設定するか "
                "vendor/ComfyUI に配置してください（README 参照）。"
            )

        config.ensure_model_dirs()
        extra_paths = write_extra_model_paths(self.paths)
        self._reap_stale_backend(log)
        self.port = _free_port(self.port)

        cmd = [
            str(self.paths.backend_python),
            str(comfy / "main.py"),
            "--listen", self.host,
            "--port", str(self.port),
            "--extra-model-paths-config", str(extra_paths),
            "--output-directory", str(self.paths.user_data / "output"),
            "--preview-method", "auto",  # stream latent previews over the ws
            "--disable-auto-launch",
            # ComfyUI v0.36+ は NVMe を検出すると重みをディスク直読み
            # （fast_disk）で扱うが、Windows でこの経路は text encoder の
            # 読み込み（bf16/fp8 -> fp16 変換時）で access violation を起こし
            # プロセスごと落ちる（anima / krea2 の TE で再現、v0.37.0）。
            # 従来どおり RAM 経由で読む。
            "--disable-fast-disk",
            # ComfyUI のアセット DB は使わない。メモリ上にすると comfyui.db の
            # ファイルロックが無くなり、残留/多重起動時の「Database is
            # locked」とその待ちが起きない。
            "--database-url", "sqlite:///:memory:",
        ]
        if self.use_ck_attention:
            # 対応可否は設定を ON にした時点で確認済み（setup.ck_attention_available）。
            # 未対応環境で付けると ComfyUI は exit(-1) するので、起動失敗時は
            # 設定で OFF にすれば復旧できる。
            cmd.append("--use-ck-attention")
            log("Comfy Kitchen INT8 attention を有効化して起動します")
        elif self.use_sage_attention:
            # パッケージが実在するときだけフラグを付ける。無いのに付けると
            # ComfyUI は起動時に exit(-1) するため（attention.py 参照）。
            from .bootstrap.setup import sage_installed
            if sage_installed(self.paths):
                cmd.append("--use-sage-attention")
                log("SageAttention を有効化して起動します")
            else:
                log("SageAttention が未インストールのため無効で起動します"
                    "（「設定…」から導入できます）")
        (self.paths.user_data / "output").mkdir(parents=True, exist_ok=True)
        log(f"ComfyUI を起動中: {' '.join(cmd)}")

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(comfy),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        # scom が死んだら OS にバックエンドを道連れにさせる + 次回起動時の
        # 掃除用に PID を記録する。
        self._job = _attach_kill_on_close_job(self._proc)
        self._write_pid_file()
        # Drain stdout on a daemon thread so a quiet subprocess never blocks the
        # readiness poll below.
        self._start_log_reader(log)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                time.sleep(0.2)  # let the reader flush the final lines
                raise BackendError(
                    f"ComfyUI が早期終了しました (code {self._proc.returncode})。\n"
                    + "\n".join(self._log_tail)
                )
            if self._ping():
                log(f"ComfyUI 準備完了: {self.base_url}")
                return
            time.sleep(0.4)
        self.stop()
        raise BackendError("制限時間内に ComfyUI が起動しませんでした")

    def _start_log_reader(self, log: Callable[[str], None]) -> None:
        def reader() -> None:
            assert self._proc and self._proc.stdout
            for raw in self._proc.stdout:
                raw = raw.rstrip("\r\n")
                clean = strip_ansi(raw).rstrip()
                if clean:
                    self._log_tail.append(clean)  # plain text for error messages
                    log(raw)                       # raw (with ANSI) for colored display

        self._log_thread = threading.Thread(target=reader, daemon=True)
        self._log_thread.start()

    def _ping(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/system_stats", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            if sys.platform == "win32":
                # Kill the whole process TREE. uv-managed venvs use a
                # trampoline python.exe that launches the real interpreter as
                # a child; terminate() alone kills only the trampoline and
                # orphans the actual ComfyUI process (which then keeps the
                # SQLite DB locked for every later launch).
                subprocess.run(
                    ["taskkill", "/PID", str(self._proc.pid), "/T", "/F"],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._job is not None and sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.CloseHandle(self._job)  # 残党も道連れ
            self._job = None
        try:
            self._pid_file.unlink(missing_ok=True)
        except OSError:
            pass

    # ----- generation ------------------------------------------------------
    def _post_prompt(self, graph: dict) -> str:
        """Queue the graph and return its prompt id.

        The id is generated here (ComfyUI accepts a client-supplied UUID) so a
        connection dropped mid-POST can be retried safely: we first ask the
        backend whether that id was already queued, and only re-send when it
        was not. Without this a retry could run the same prompt twice.
        """
        prompt_id = str(uuid.uuid4())
        payload = json.dumps({"prompt": graph, "client_id": self.client_id,
                              "prompt_id": prompt_id}).encode()
        last: Exception = BackendError("no attempt was made")
        for n in range(_HTTP_ATTEMPTS):
            req = urllib.request.Request(
                self.base_url + "/prompt", data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                return str(data.get("prompt_id") or prompt_id)
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")
                raise BackendError(f"prompt が拒否されました: {detail}") from e
            except _RETRY_ERRORS as e:
                last = e
                if self._prompt_queued(prompt_id):
                    return prompt_id     # 届いていた: 再送しない
                time.sleep(min(_HTTP_BACKOFF * (2 ** n), 4.0))
        raise BackendError(
            f"バックエンドへの接続に失敗しました（{_HTTP_ATTEMPTS} 回再試行）: {last}"
        ) from last

    def _prompt_queued(self, prompt_id: str) -> bool:
        """True when the backend already knows this prompt (queued or done)."""
        try:
            if json.loads(
                self._get(self.base_url + f"/history/{prompt_id}", attempts=2)
            ).get(prompt_id):
                return True
            queue = json.loads(self._get(self.base_url + "/queue", attempts=2))
        except (BackendError, ValueError, urllib.error.HTTPError):
            return False
        for key in ("queue_running", "queue_pending"):
            for item in queue.get(key, []):
                if len(item) > 1 and item[1] == prompt_id:
                    return True
        return False

    def _get(self, url: str, timeout: float = 60.0,
             attempts: int = _HTTP_ATTEMPTS) -> bytes:
        """GET with retries for the transient failures the backend shows right
        after a prompt finishes.

        ComfyUI does its post-execution work (GC / VRAM staging / output
        registration) on the worker thread, which holds the GIL and starves the
        aiohttp event loop. Requests issued in that window get the connection
        reset (WinError 10054) or time out even though the server is healthy.
        Retrying a few times rides it out; only GETs are retried (they are
        idempotent — the prompt POST must never be replayed).
        """
        last: Exception = BackendError("no attempt was made")
        for n in range(attempts):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError:
                raise                      # 404 等はリトライしても無駄
            except _RETRY_ERRORS as e:
                last = e
                time.sleep(min(_HTTP_BACKOFF * (2 ** n), 4.0))
        raise BackendError(
            f"バックエンドへの接続に失敗しました（{attempts} 回再試行）: {last}"
        ) from last

    def _fetch_image(self, filename: str, subfolder: str, ftype: str) -> bytes:
        qs = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": ftype}
        )
        return self._get(self.base_url + "/view?" + qs)

    def _history_images(self, prompt_id: str) -> list[bytes]:
        history = json.loads(self._get(self.base_url + f"/history/{prompt_id}"))
        entry = history.get(prompt_id, {})
        images: list[bytes] = []
        for node_out in entry.get("outputs", {}).values():
            for img in node_out.get("images", []):
                images.append(
                    self._fetch_image(img["filename"], img.get("subfolder", ""),
                                      img.get("type", "output"))
                )
        return images

    def generate(self, graph: dict,
                 on_progress: Optional[Callable[[Progress], None]] = None,
                 on_preview: Optional[Callable[[bytes], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None,
                 on_cached: Optional[Callable[[list], None]] = None,
                 on_timing: Optional[Callable[[float], None]] = None) -> list[bytes]:
        """Run a graph to completion and return output PNG bytes.

        ``on_progress`` receives Progress updates; ``on_preview`` receives raw
        JPEG/PNG bytes of intermediate latent previews; ``cancel`` is polled
        and, if it returns True, the run is interrupted. ``on_cached`` receives
        the node ids served from the backend's output cache (sent once at the
        start of execution) — the app uses it to tell whether a merged model
        was still in RAM or had to be rebuilt. ``on_timing`` receives, at
        completion, the pure inference time in seconds — the wall time the
        backend spent executing sampler (KSampler) nodes only, excluding model
        loading / text encoding / VAE decode.
        """
        if not self.is_running():
            raise BackendError("バックエンドが起動していません")
        on_progress = on_progress or (lambda _p: None)
        on_preview = on_preview or (lambda _b: None)
        cancel = cancel or (lambda: False)
        on_cached = on_cached or (lambda _n: None)

        # 推論時間 = サンプラーノードが「実行中」だった時間の合計。
        # executing イベントはノード開始時に飛ぶので、サンプラーに入った時刻
        # から次のノードへ移った時刻までを積算する。
        sampler_nodes = {nid for nid, node in graph.items()
                         if node.get("class_type") == "KSampler"}
        sample_secs = 0.0
        sample_enter: Optional[float] = None
        ws_alive = True          # False = 接続断 -> HTTP ポーリングへ切替

        ws = websocket.WebSocket()
        ws.connect(
            f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}", timeout=10
        )
        ws.settimeout(1.0)

        prompt_id = self._post_prompt(graph)
        try:
            while True:
                if cancel():
                    self.interrupt()
                    raise BackendError("生成をキャンセルしました")
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except (websocket.WebSocketException, OSError):
                    # 生成直後などにサーバのイベントループが詰まると接続を
                    # 切られることがある（WinError 10054）。プロンプト自体は
                    # 実行され続けるので、以降は HTTP ポーリングで完了を待つ。
                    ws_alive = False
                    break
                if isinstance(msg, (bytes, bytearray)):
                    # Binary frame: [4B event][4B image format][image bytes].
                    # event 1 == PREVIEW_IMAGE.
                    if len(msg) > 8 and int.from_bytes(msg[:4], "big") == 1:
                        on_preview(bytes(msg[8:]))
                    continue
                event = json.loads(msg)
                etype = event.get("type")
                data = event.get("data", {})
                if etype == "progress":
                    on_progress(Progress(
                        value=data.get("value", 0),
                        maximum=data.get("max", 0),
                        note="サンプリング",
                    ))
                elif etype == "execution_cached":
                    if data.get("prompt_id") == prompt_id:
                        on_cached(list(data.get("nodes", [])))
                elif etype == "executing":
                    if data.get("prompt_id") != prompt_id:
                        continue
                    node = data.get("node")
                    now = time.monotonic()
                    if sample_enter is not None and node not in sampler_nodes:
                        sample_secs += now - sample_enter
                        sample_enter = None
                    elif sample_enter is None and node in sampler_nodes:
                        sample_enter = now
                    if node is None:
                        break  # finished
                elif etype == "execution_error" and data.get("prompt_id") == prompt_id:
                    raise BackendError(
                        f"実行エラー: {data.get('exception_message', data)}"
                    )
        finally:
            try:
                ws.close()
            except Exception:
                pass

        if not ws_alive:
            # 進捗は取りこぼすが、完了は履歴で確認できる。
            self._wait_prompt(prompt_id, cancel)
        elif on_timing is not None and sample_secs > 0:
            on_timing(sample_secs)   # 途中で切れた計測値は使わない
        return self._history_images(prompt_id)

    def generate_text(self, graph: dict,
                      cancel: Optional[Callable[[], bool]] = None,
                      timeout: float = 900.0) -> str:
        """文字列を出力するグラフ（TextGenerate -> PreviewAny 等）を実行し、
        履歴の "text" 出力を連結して返す。進捗は不要なので websocket は
        使わず /history のポーリングで完了を待つ。"""
        if not self.is_running():
            raise BackendError("バックエンドが起動していません")
        cancel = cancel or (lambda: False)
        prompt_id = self._post_prompt(graph)
        self._wait_prompt(prompt_id, cancel, timeout=timeout)
        history = json.loads(self._get(self.base_url + f"/history/{prompt_id}"))
        entry = history.get(prompt_id, {})
        texts: list[str] = []
        for node_out in entry.get("outputs", {}).values():
            for t in node_out.get("text", []) or []:
                texts.append(str(t))
        return "\n".join(texts)

    def _wait_prompt(self, prompt_id: str, cancel: Callable[[], bool],
                     timeout: float = 1800.0) -> None:
        """Wait for ``prompt_id`` to finish by polling /history.

        Used when the progress websocket drops mid-run: the backend keeps
        executing the queued prompt, so the history endpoint is the source of
        truth for completion.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cancel():
                self.interrupt()
                raise BackendError("生成をキャンセルしました")
            entry = json.loads(
                self._get(self.base_url + f"/history/{prompt_id}")
            ).get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise BackendError(f"実行エラー: {status.get('messages')}")
                if entry.get("outputs") or status.get("completed"):
                    return
            time.sleep(0.5)
        raise BackendError("生成の完了を確認できませんでした")

    def interrupt(self) -> None:
        try:
            req = urllib.request.Request(self.base_url + "/interrupt", data=b"")
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    def free_memory(self) -> None:
        """Ask the backend to drop ALL cached node outputs and loaded models.

        ComfyUI has no per-entry cache eviction, so this is all-or-nothing;
        anything still needed is rebuilt/reloaded on next use. (Pinned merged
        models are separate — see release_merge/release_all_merges.)
        """
        payload = json.dumps({"unload_models": True, "free_memory": True}).encode()
        req = urllib.request.Request(
            self.base_url + "/free", data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)

    # ----- scom merge pin cache (routes served by our custom node) ---------
    def merge_pinned(self) -> list[str]:
        """Pin-cache keys of merged models currently held in backend RAM."""
        with urllib.request.urlopen(self.base_url + "/scom/merges",
                                    timeout=5) as resp:
            return list(json.loads(resp.read()).get("pinned", []))

    def _post_merge_release(self, payload: dict) -> int:
        req = urllib.request.Request(
            self.base_url + "/scom/merge_release",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(json.loads(resp.read()).get("released", 0))

    def release_merge(self, recipe: str, quantize: str,
                      low_memory: bool) -> int:
        """Free one pinned merged model from backend RAM."""
        return self._post_merge_release({
            "recipe": recipe, "quantize": quantize,
            "low_memory": bool(low_memory)})

    def release_all_merges(self) -> int:
        """Free every pinned merged model (and cached difference LoRA) from
        backend RAM."""
        return self._post_merge_release({"all": True})

    def cached_loras(self) -> list[str]:
        """Keys of the difference LoRAs currently held in backend RAM."""
        with urllib.request.urlopen(self.base_url + "/scom/loras",
                                    timeout=5) as resp:
            return list(json.loads(resp.read()).get("cached", []))

    def release_cached_lora(self, model_a: str, model_b: str,
                            rank: int) -> int:
        """Free one cached difference LoRA (inputs as in the graph)."""
        req = urllib.request.Request(
            self.base_url + "/scom/lora_release",
            data=json.dumps({"model_a": model_a, "model_b": model_b,
                             "rank": int(rank)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(json.loads(resp.read()).get("released", 0))

    def object_info(self, class_type: str) -> dict:
        """Fetch node metadata (used to discover valid sampler/clip options)."""
        with urllib.request.urlopen(
            self.base_url + f"/object_info/{class_type}", timeout=10
        ) as resp:
            return json.loads(resp.read())
