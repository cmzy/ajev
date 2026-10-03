#!/usr/bin/env python3
"""通过 Google 官方的 `colab` CLI，在 Colab VM 上驱动 AJev 的训练。

本地写代码、远程 GPU 训练：这个脚本把“开 VM → 上传代码和数据 → 安装 → 后台训练 →
查看进度 → 下载结果 → 释放 VM”这一整套操作封装成几个子命令。它对应 AJev 流程中的
“训练”环节（数据在本地由 ajev.data.build 构建好后上传）。

用法::

    python scripts/colab_job.py up      -s ajev --gpu T4 --data data/build   # 开 VM，上传代码和数据，安装
    python scripts/colab_job.py drive   -s ajev                              # 把 Google Drive 挂载到 /content/drive
    python scripts/colab_job.py train   -s ajev --run sft1 -- --epochs 3 --grad-ckpt
    python scripts/colab_job.py status  -s ajev --run sft1                   # GPU 状态 + 最后几行日志
    python scripts/colab_job.py fetch   -s ajev --run sft1 --what best       # 下载 runs/<run>/<what>
    python scripts/colab_job.py stop    -s ajev

训练进程在 VM 上以脱离会话的方式在后台运行（类似 nohup），本地终端断开也不影响训练。
加 ``--drive`` 时，运行目录放在 Google Drive 上，VM 被回收也不会丢失 checkpoint
（前提是你的账号能成功挂载 Drive；否则就定期 ``fetch --what last`` 下载到本地，
或者给训练脚本传 ``--hub-repo`` 上传到 HF Hub）。用同一个 ``--run`` 再次执行 ``train``，
会从该运行目录下的 ``last`` checkpoint 断点续训。

几个踩过的坑：

* 代码要以文件形式发给 VM（``colab exec -f``）；通过 stdin 管道传代码会一直卡住；
* ``colab drivemount`` 第一次使用时需要真人在终端里打开授权链接并按回车，
  授权过一次之后，非交互调用也能挂载成功；
* 挂载失败时 ``drivemount`` 不一定返回非零退出码，所以 ``drive`` 子命令会再检查一次是否真的挂上；
* websocket 偶尔会断（"Connection was lost"），远程执行会自动重试。

------------------------------------------------------------------------------
给初学者的背景知识
------------------------------------------------------------------------------

【VM 与会话（session）】
VM（虚拟机）就是 Google 数据中心里临时租给你的一台带 GPU 的 Linux 电脑。``colab new`` 申请一台，
``colab stop`` 归还。每台 VM 上跑着一个 Jupyter kernel（Python 解释器进程），``colab exec``
就是把代码发给这个 kernel 执行，效果和在 Colab 网页的代码格子里运行一样。
“会话”是本地给这台 VM 起的名字（``-s ajev``），后续命令靠它找到同一台 VM。
注意：VM 上的文件是临时的，VM 被归还或被 Google 回收后，/content 下的东西全部消失。

【后台进程 / nohup】
训练要跑好几个小时，而 ``colab exec`` 只是一次短连接。如果让训练在 exec 里直接运行，
连接一断训练就跟着停了。所以这里用 ``subprocess.Popen`` 在 VM 上另起一个独立进程跑训练，
并让它脱离当前会话（start_new_session=True，效果类似 Linux 的 nohup 命令）：
exec 立刻返回，训练在后台继续跑，日志写到文件里，之后用 ``status`` 去看。

【僵尸进程】
在 Linux 里，子进程结束后，要等父进程“回收”（读取它的退出状态）才会彻底消失；
在被回收之前，它会以“僵尸”（状态 Z）的形式留在进程表里。我们的训练进程是 kernel 的子进程，
kernel 不会主动回收它，所以训练崩溃后 /proc/<pid> 目录仍然存在，必须读状态字段才能判断它其实已经死了。

【tar 打包上传】
代码目录里有很多小文件，一个个上传很慢。tar.gz 把整个目录打包并压缩成一个文件，
上传一次，再在 VM 上解压，目录结构保持不变。

【Google Drive 挂载】
“挂载”就是把你的 Google Drive 变成 VM 上的一个文件夹（/content/drive/MyDrive）。
往这个文件夹里写文件，就等于写进了你的 Drive，VM 被回收后文件依然在 Drive 里，
所以 checkpoint 放在这里，就能在新 VM 上断点续训。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time

# 本地仓库根目录（本文件位于 <ROOT>/scripts/ 下）。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# VM 上存放代码和数据的目录；/content 是 Colab 的默认工作目录。
REMOTE = "/content/AJev"
# 挂载 Drive 后，运行目录（checkpoint、日志）所在位置，对应你 Drive 里的 MyDrive/ajev/runs/。
DRIVE_RUNS = "/content/drive/MyDrive/ajev/runs"
# 需要上传到 VM 的代码路径（相对仓库根目录）；数据单独打包上传。
CODE_PATHS = ["ajev", "scripts", "pyproject.toml", "README.md"]


def colab(*args: str, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    """调用本地的 ``colab`` 命令行，并先打印出要执行的完整命令，方便排查。

    ``check=True`` 时命令返回非零退出码会抛出 ``CalledProcessError``；
    需要自己处理失败（例如重试）时传 ``check=False``。

    举个例子：colab("stop", "-s", "ajev")
        → 先打印 "$ colab stop -s ajev"，再真正执行这条命令。
    ``*args`` 表示把任意多个位置参数收集成一个元组；``shlex.quote`` 会给含空格等特殊字符的参数加引号，
    这样打印出来的命令可以直接复制到终端里运行。
    ``timeout``（秒）：超时就结束这条命令，返回退出码 124（与 shell 的 timeout 命令一致），而不是一直卡住。
    """
    print("$ colab " + " ".join(shlex.quote(a) for a in args), flush=True)
    try:
        return subprocess.run(["colab", *args], check=check, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[colab_job] command timed out after {timeout}s", flush=True)
        if check:
            raise
        return subprocess.CompletedProcess(["colab", *args], 124)


def remote_python(session: str, code: str, retries: int = 3, timeout: float = 600) -> None:
    """在 VM 的 Jupyter kernel 里执行一段 Python 代码。

    做法：把代码（先用 ``textwrap.dedent`` 去掉公共缩进）写进本地临时 .py 文件，
    再用 ``colab exec -f`` 发送执行——通过 stdin 管道传代码会卡住，所以必须走文件。

    kernel 的 websocket 偶尔会断开（报 "Connection was lost"），通常发生在空闲一段时间后的
    第一次调用，因此失败时最多重试 ``retries`` 次，每次间隔 5 秒；全部失败则退出。
    注意：重试意味着同一段代码可能被执行多次，所以发过去的代码应当是幂等的
    （例如 ``train`` 会先检查是否已有训练进程在跑）。
    “幂等”的意思是：执行一次和执行多次的效果相同。

    处理步骤：
        第 1 步：写临时文件（delete=False 让文件在 with 结束后仍保留，供 colab exec 读取）；
        第 2 步：最多尝试 retries 次 colab exec，成功（退出码 0）就返回；每次最多等 ``timeout`` 秒
                （连接偶尔会挂住不返回，没有超时就会一直等下去：曾经因此白等了半小时）；
        第 3 步：无论成功失败，finally 里都删除临时文件，不在本地留垃圾。

    举个例子：remote_python("ajev", "print(1 + 1)") → VM 上执行后，本地终端打印 2。
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        path = f.name
    try:
        for attempt in range(1, retries + 1):
            r = colab("exec", "-s", session, "-f", path, check=False, timeout=timeout)
            if r.returncode == 0:
                return
            print(f"[colab_job] exec failed (attempt {attempt}/{retries})", flush=True)
            time.sleep(5)
        raise SystemExit("colab exec kept failing")
    finally:
        os.unlink(path)


def upload_file(session: str, local: str, remote: str, chunk_mb: int = 40) -> None:
    """把本地文件上传到 VM；大文件分块上传后在 VM 上拼回，并核对 SHA-256。

    为什么要分块：Colab 的上传接口对单个文件的大小有限制，约 100 MB 的数据包会直接返回 500 错误
    （第二版数据压缩后就有这么大）。做法：
        第 1 步：本地按 chunk_mb 切成若干块，逐块上传为 <remote>.part000、.part001……；
        第 2 步：在 VM 上按顺序拼接成 <remote>，计算 SHA-256 和本地的比较，一致后才删除分块；
        第 3 步：这一步可以安全地重复执行（远程执行超时重试时）：分块已经删掉、但拼好的文件在且校验一致，就直接算成功。
    小于 chunk_mb 的文件直接整个上传。
    """
    import hashlib

    size = os.path.getsize(local)
    if size <= chunk_mb * 2**20:
        colab("upload", "-s", session, local, remote)
        return
    h = hashlib.sha256()
    parts = []
    with open(local, "rb") as f:
        k = 0
        while chunk := f.read(chunk_mb * 2**20):
            h.update(chunk)
            # 每次上传用独立的临时文件名（两个上传同时进行时曾因共用 upload.partNNN 互相覆盖）
            fd, part_local = tempfile.mkstemp(prefix=f"upload_{os.getpid()}_", suffix=f".part{k:03d}")
            os.close(fd)
            with open(part_local, "wb") as out:
                out.write(chunk)
            part_remote = f"{remote}.part{k:03d}"
            colab("upload", "-s", session, part_local, part_remote)
            os.remove(part_local)
            parts.append(part_remote)
            k += 1
    remote_python(session, f"""
        import hashlib, os
        parts, remote, want = {parts!r}, {remote!r}, {h.hexdigest()!r}
        if all(os.path.exists(p) for p in parts):
            h = hashlib.sha256()
            with open(remote, "wb") as out:
                for p in parts:
                    data = open(p, "rb").read()
                    h.update(data)
                    out.write(data)
            ok = h.hexdigest() == want
        else:  # 上一次尝试已经拼好并删除了分块：直接校验拼好的文件
            h = hashlib.sha256()
            with open(remote, "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
            ok = h.hexdigest() == want
        if not ok:
            raise SystemExit("SHA-256 mismatch after reassembling " + remote)
        for p in parts:
            if os.path.exists(p):
                os.remove(p)
        print("reassembled", remote, os.path.getsize(remote), "bytes, sha256 ok")
    """)


def make_tar(paths: list[str], base: str, name: str) -> str:
    """把 ``base`` 下的若干路径打包成系统临时目录中的 ``name``（tar.gz），返回压缩包路径。

    包内路径保持相对 ``base`` 的结构（arcname），解压到 VM 的 REMOTE 目录后与本地仓库布局一致；
    ``__pycache__`` 会被过滤掉，减少上传体积。

    举个例子：make_tar(["ajev", "scripts"], ROOT, "ajev_code.tar.gz")
        → 生成 /tmp/.../ajev_code.tar.gz，包内是 ajev/... 和 scripts/... 两个目录。
    ``filter`` 参数是一个函数：对包里的每个文件调用一次，返回 None 表示“不要这个文件”，
    原样返回 ti 表示保留。这里用 lambda 写成一行。
    """
    # 每次打包放在独立的临时目录（两个上传同时进行时曾因共用同名压缩包互相覆盖、解压出错）
    out = os.path.join(tempfile.mkdtemp(prefix="colab_job_"), name)
    with tarfile.open(out, "w:gz") as tar:
        for p in paths:
            tar.add(os.path.join(base, p), arcname=p,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return out


def runs_dir(drive: bool) -> str:
    """返回运行目录的根：``--drive`` 时在 Google Drive 上（VM 回收后仍在），否则在 VM 本地磁盘上。"""
    return DRIVE_RUNS if drive else f"{REMOTE}/runs"


def cmd_up(a) -> None:
    """``up`` 子命令：准备好一台可以训练的 VM。

    1. 若同名会话不存在则新建（可指定 GPU 类型，默认 T4）；已存在则复用，只重新上传代码；
    2. 打包上传代码；如果传了 ``--data``，把本地数据目录也打包上传；
    3. 在 VM 上解压到 REMOTE，并以 ``pip install --no-deps -e`` 安装 ajev 包，
       最后打印 torch / transformers 版本和 GPU 信息，确认环境可用。

    “可编辑安装”（pip install -e）：不把代码复制进 site-packages，而是让 Python 直接从
    /content/AJev 目录导入 ajev 包。以后重新上传代码、覆盖这个目录，就能立刻生效，不用重装。

    举个例子：python scripts/colab_job.py up -s ajev --gpu T4 --data data/build
        → 新开一台 T4（若 ajev 会话不存在），上传代码和 data/build，安装后打印类似
          "torch 2.11.0 transformers 5.16.1 cuda True Tesla T4"。
    """
    # 第 1 步：用会话列表输出中是否出现 "[会话名]" 来判断会话是否已存在，不存在才新开 VM。
    # capture_output=True 把命令输出收集成字符串（而不是打印到屏幕），text=True 表示按文本解码。
    sessions = subprocess.run(["colab", "sessions"], capture_output=True, text=True).stdout
    if f"[{a.session}]" not in sessions:
        colab("new", "-s", a.session, *(["--gpu", a.gpu] if a.gpu else []))
    # 第 2 步：打包并上传代码；有 --data 时再打包上传数据目录。
    code = make_tar(CODE_PATHS, ROOT, "ajev_code.tar.gz")
    colab("upload", "-s", a.session, code, "/content/ajev_code.tar.gz")
    if a.data:
        data = make_tar([os.path.relpath(a.data, ROOT)], ROOT, "ajev_data.tar.gz")
        upload_file(a.session, data, "/content/ajev_data.tar.gz")
    # 第 3 步（远程执行）：解压并删除压缩包；Colab 已预装 torch / transformers / datasets，
    # 用 --no-deps 只安装 ajev 本身，避免 pip 改动 Colab 自带的依赖版本。
    remote_python(a.session, f"""
        import os, subprocess, tarfile
        os.makedirs("{REMOTE}", exist_ok=True)
        for t in ("/content/ajev_code.tar.gz", "/content/ajev_data.tar.gz"):
            if os.path.exists(t):
                tarfile.open(t).extractall("{REMOTE}")
                os.remove(t)
        # Colab already ships torch / transformers / datasets; install only our package.
        r = subprocess.run(["pip", "install", "-q", "--no-deps", "-e", "{REMOTE}"], capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])
        import torch, transformers
        print("torch", torch.__version__, "transformers", transformers.__version__,
              "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
        print(os.listdir("{REMOTE}"))
    """)


def cmd_drive(a) -> None:
    """``drive`` 子命令：在 VM 上挂载 Google Drive，并确认挂载成功。

    第一次使用需要你本人在终端里运行 ``colab drivemount`` 完成授权（要打开链接并按回车），
    之后本命令可以非交互地挂载。挂载后会再用 ``os.path.ismount`` 检查一次：
    ``drivemount`` 失败时不一定返回非零退出码，如果不检查，后面 ``os.makedirs`` 会在 VM
    本地磁盘上建出一个同名目录，看起来像存进了 Drive，实际上 VM 一回收就全丢了。

    ``os.path.ismount(路径)`` 判断该路径是不是一个真正的挂载点（另一个文件系统接入的位置），
    普通文件夹返回 False，所以可以用来区分“真的 Drive”和“本地同名目录”。

    处理步骤：
        第 1 步：执行 colab drivemount；
        第 2 步：远程检查 /content/drive 是否真的是挂载点，不是就报错退出；
        第 3 步：在 Drive 上创建运行目录根（MyDrive/ajev/runs）并列出已有的运行。
    """
    colab("drivemount", "-s", a.session)
    remote_python(a.session, f"""
        import os
        # drivemount can fail without a non-zero exit; never create the path on local disk by mistake.
        if not os.path.ismount("/content/drive"):
            raise SystemExit("Google Drive is NOT mounted; use local runs + `fetch`, or --hub-repo")
        os.makedirs("{DRIVE_RUNS}", exist_ok=True)
        print("drive runs dir:", "{DRIVE_RUNS}", os.listdir("{DRIVE_RUNS}"))
    """)


def cmd_train(a) -> None:
    """``train`` 子命令：在 VM 上后台启动 ``python -m ajev.train.train``。

    * 训练集 / 验证集固定取 VM 上 ``<REMOTE>/<data_dir>`` 下的 train.jsonl、val.jsonl、val_typed.jsonl，
      ``--`` 之后的参数原样转发给训练脚本；
    * 输出目录为 ``<runs_dir>/<run>``；目录里已有 ``last`` checkpoint 时训练脚本会自动续训；
    * 启动前检查 pid 文件：若该运行目录已有训练进程在跑（且不是僵尸进程）就拒绝再启动一个，
      防止 ``remote_python`` 重试或误操作导致两个进程同时写同一个目录；
    * 用 ``start_new_session=True`` 让训练进程脱离 kernel 的会话，标准输出 / 错误写入
      ``stdout.log``，进程号写入 ``pid`` 文件，供 ``status`` 查询。

    举个例子：
        python scripts/colab_job.py train -s ajev --run sft1 --drive -- --epochs 2 --grad-ckpt
        → 在 VM 上后台执行：
          python -m ajev.train.train --train /content/AJev/data/build/train.jsonl
              --val /content/AJev/data/build/val.jsonl /content/AJev/data/build/val_typed.jsonl
              --out /content/drive/MyDrive/ajev/runs/sft1 --epochs 2 --grad-ckpt

    处理步骤：
        第 1 步：确定输出目录和数据目录，拼出完整的训练命令；
        第 2 步（远程）：创建输出目录，检查是否已有活着的训练进程，有就拒绝启动；
        第 3 步（远程）：以追加模式打开 stdout.log，用 Popen 在后台启动训练；
        第 4 步（远程）：把进程号写入 pid 文件，打印启动信息后立即返回（不等训练结束）。

    远程代码里的 ``{out!r}`` 是 f-string 的写法：!r 表示用 repr() 插入，
    字符串会自带引号，生成的远程代码才是合法的 Python。
    """
    # 第 1 步：拼出输出目录、数据目录和训练命令行参数。
    out = f"{runs_dir(a.drive)}/{a.run}"
    data = f"{REMOTE}/{a.data_dir}"
    args = ["--train", f"{data}/train.jsonl", "--val", f"{data}/val.jsonl", f"{data}/val_typed.jsonl",
            "--out", out, *a.train_args]
    # 用 shlex.quote 逐个转义参数后拼成一条 shell 命令，参数里有空格或特殊字符也安全。
    cmd = " ".join(shlex.quote(x) for x in ["python", "-m", "ajev.train.train", *args])
    remote_python(a.session, f"""
        import os, subprocess
        os.makedirs({out!r}, exist_ok=True)
        pid_file = {out!r} + "/pid"
        if os.path.exists(pid_file):
            stat = "/proc/" + open(pid_file).read().strip() + "/stat"
            if os.path.exists(stat) and open(stat).read().split()[2] != "Z":
                raise SystemExit("training already running for this run dir; not starting another")
        log = open({out!r} + "/stdout.log", "a")
        p = subprocess.Popen({cmd!r}, shell=True, cwd="{REMOTE}", stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        open({out!r} + "/pid", "w").write(str(p.pid))
        print("started pid", p.pid, "->", {out!r})
    """)


def cmd_status(a) -> None:
    """``status`` 子命令：打印 GPU 利用率 / 显存、训练进程是否还活着，以及 stdout.log 的最后 ``--lines`` 行。

    判断进程存活时除了看 ``/proc/<pid>`` 是否存在，还要看进程状态是不是 "Z"（僵尸）：
    训练进程是 kernel 的子进程，崩溃后若父进程没有回收，会以僵尸状态残留在 /proc 中，
    只看目录是否存在会误报为“运行中”。

    ``/proc/<pid>/stat`` 是 Linux 提供的进程信息文件，内容用空格分隔，第 3 个字段就是进程状态
    （R=运行、S=睡眠、Z=僵尸……），所以代码里取 ``split()[2]``。

    举个例子：python scripts/colab_job.py status -s ajev --run sft1 --drive --lines 5
        → 输出类似：
          99 %, 8559 MiB, 15360 MiB
          pid 1657 running
          {"step": 20, "epoch": 0.014, "loss": 1.5887, ...}
          ...（最后 5 行日志）

    远程代码中的 ``{{pid}}``：外层是本地的 f-string，双花括号表示“输出一个字面的 {”，
    这样发到 VM 上的代码里才是 ``f"/proc/{pid}/stat"``，由 VM 上的 Python 再去替换。
    """
    out = f"{runs_dir(a.drive)}/{a.run}"
    remote_python(a.session, f"""
        import os, subprocess
        print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
        pid = open({out!r} + "/pid").read().strip() if os.path.exists({out!r} + "/pid") else None
        stat = f"/proc/{{pid}}/stat"
        # A crashed child stays a zombie ("Z") under the kernel process, so check the state too.
        alive = pid is not None and os.path.exists(stat) and open(stat).read().split()[2] != "Z"
        print("pid", pid, "running" if alive else "NOT running")
        lines = open({out!r} + "/stdout.log").read().splitlines() if os.path.exists({out!r} + "/stdout.log") else []
        print("\\n".join(l[:300] for l in lines[-{a.lines}:]))
    """)


def cmd_fetch(a) -> None:
    """``fetch`` 子命令：把 VM 上 ``<runs_dir>/<run>/<what>``（默认 ``best``）下载到本地 ``runs/<run>/``，并校验完整性。

    为什么要校验：我们吃过亏——Google Drive 在后台异步上传大文件，VM 被回收时新版本没传完，
    云端留下的是旧文件，看起来“保存成功”，实际权重是错的。所以现在只相信“下载到本机并核对过哈希”的副本。

    处理步骤：
        第 1 步（远程）：把 checkpoint 目录和日志打包成 tar（不压缩：模型权重几乎压不动，压缩只会浪费时间），
                         并计算整个 tar 的 SHA-256 写进同名 .sha256 文件；
        第 2 步：下载 tar 和 .sha256 到本地；
        第 3 步：本地重新计算 tar 的 SHA-256，与远程的比较；不一致就报错，绝不使用这份文件；
        第 4 步：先解压到临时目录 runs/<run>/.incoming，再用改名替换旧的 runs/<run>/<what>，
                 这样本地任何时刻都只有完整的旧版本或完整的新版本。

    举个例子：python scripts/colab_job.py fetch -s ajev --run sft2 --what best
        → 本地得到 runs/sft2/best/（权重、tokenizer、配置）以及 runs/sft2/log.jsonl、runs/sft2/args.json。
    """
    import hashlib
    import shutil

    src = f"{runs_dir(a.drive)}/{a.run}"
    tar_remote = f"/content/fetch_{a.run}_{a.what}.tar"
    # 第 1 步（远程）：打包并计算哈希。
    remote_python(a.session, f"""
        import hashlib, tarfile
        with tarfile.open({tar_remote!r}, "w") as t:
            t.add({src!r} + "/" + {a.what!r}, arcname={a.what!r})
            for f in ("log.jsonl", "args.json"):
                try:
                    t.add({src!r} + "/" + f, arcname=f)
                except FileNotFoundError:
                    pass
        h = hashlib.sha256()
        with open({tar_remote!r}, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        open({tar_remote!r} + ".sha256", "w").write(h.hexdigest())
        print("remote sha256", h.hexdigest()[:16])
    """)
    # 第 2 步：下载 tar 和哈希文件。
    local_dir = os.path.join(ROOT, "runs", a.run)
    os.makedirs(local_dir, exist_ok=True)
    local_tar = os.path.join(local_dir, f"{a.what}.tar")
    colab("download", "-s", a.session, tar_remote, local_tar, timeout=3600)
    colab("download", "-s", a.session, tar_remote + ".sha256", local_tar + ".sha256", timeout=300)
    # 第 3 步：本地校验。
    h = hashlib.sha256()
    with open(local_tar, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    expected = open(local_tar + ".sha256").read().strip()
    if h.hexdigest() != expected:
        raise SystemExit(f"SHA-256 mismatch for {local_tar}: local {h.hexdigest()[:16]} != remote {expected[:16]}")
    # 第 4 步：解压到临时目录，再整体替换。
    incoming = os.path.join(local_dir, ".incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    with tarfile.open(local_tar) as t:
        t.extractall(incoming)
    target = os.path.join(local_dir, a.what)
    shutil.rmtree(target, ignore_errors=True)
    os.rename(os.path.join(incoming, a.what), target)
    for f in os.listdir(incoming):  # log.jsonl / args.json
        os.replace(os.path.join(incoming, f), os.path.join(local_dir, f))
    shutil.rmtree(incoming)
    os.remove(local_tar)
    os.remove(local_tar + ".sha256")
    print(f"fetched + verified (sha256 {expected[:16]}) ->", target)


def remote_json(session: str, code: str, timeout: float = 150, retries: int = 3) -> dict:
    """在 VM 上执行一段 Python，读取它打印的一行 ``SYNCJSON:{...}`` 并解析成字典（带超时和重试）。

    和 remote_python 不同，这里要拿回结果，所以捕获输出；连接挂住时最多等 timeout 秒再重试。
    全部失败就抛异常——调用方据此判断“读不到状态”，而不是当作“还没完成”继续空等。
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        path = f.name
    try:
        for attempt in range(1, retries + 1):
            try:
                r = subprocess.run(["colab", "exec", "-s", session, "-f", path], capture_output=True, text=True,
                                   timeout=timeout)
                for line in (r.stdout or "").splitlines():
                    if "SYNCJSON:" in line:
                        return json.loads(line.split("SYNCJSON:", 1)[1])
            except subprocess.TimeoutExpired:
                pass
            time.sleep(5)
        raise RuntimeError(f"could not read remote state from session {session!r}")
    finally:
        os.unlink(path)


def cmd_sync(a) -> None:
    """``sync`` 子命令：在本机后台持续运行，把 VM 上的新结果**一出现就下载**（数据审计后的防丢失措施）。

    背景：gemma_lora4 训练和评测都已完成，但 Colab 在约 8 小时后终止了 VM，而“等全部结束再下载”的任务
    又因为读取脚本写错一直空转，最终模型和评测结果全部丢失。所以：
        1. checkpoint 一保存就下载：查询 VM 上 ``<run>/last`` 的步数，比本地新就立即 fetch（sha256 校验）；
           训练日志出现 "done" 后再下载最终的 ``best``；
        2. 评测结果逐个下载：``--files`` 给出的文件（可用通配符）一出现、且大小两次查询不变（写完了），就下载并核对大小；
        3. 启动前自检：先实际查询一次远程状态，读不到就立即报错退出，绝不悄悄空转；
           运行中连续多次读不到状态（VM 可能已被终止）也报错退出。
    ``--done-file``：VM 上出现这个文件（例如评测脚本最后写的完成标记）且所有文件都已下载后，正常退出。
    """
    run_dir = f"{runs_dir(a.drive)}/{a.run}"
    probe = f"""
        import glob, json, os
        out = {{"files": {{}}}}
        try:
            out["last_step"] = json.load(open({run_dir + "/last/ajev_lm_config.json"!r}))["step"]
        except Exception:
            out["last_step"] = None
        log = {run_dir + "/log.jsonl"!r}
        out["done"] = os.path.exists(log) and '"done": true' in open(log).read()
        out["best"] = os.path.exists({run_dir + "/best/adapter_model.safetensors"!r})
        for pat in {list(a.files)!r}:
            for p in glob.glob("/content/AJev/" + pat):
                out["files"][p[len("/content/AJev/"):]] = os.path.getsize(p)
        out["done_file"] = bool({a.done_file!r}) and os.path.exists("/content/AJev/" + ({a.done_file!r} or ""))
        print("SYNCJSON:" + json.dumps(out))
    """
    local_run = os.path.join(ROOT, "runs", a.run)
    os.makedirs(local_run, exist_ok=True)

    def local_step() -> int | None:
        try:
            with open(os.path.join(local_run, "last", "ajev_lm_config.json")) as f:
                return json.load(f)["step"]
        except (OSError, ValueError, KeyError):
            return None

    # 第 3 项：启动自检
    try:
        state = remote_json(a.session, probe)
    except RuntimeError as e:
        raise SystemExit(f"[sync] self-test failed: {e}. Not starting (nothing would be downloaded).")
    print(f"[sync] self-test ok: remote last step {state['last_step']}, {len(state['files'])} matching files", flush=True)
    sizes: dict[str, int] = {}
    got: dict[str, int] = {}
    best_done = False
    failures = 0
    while True:
        try:
            state = remote_json(a.session, probe)
            failures = 0
        except RuntimeError as e:
            failures += 1
            print(f"[sync] {time.strftime('%H:%M')} {e} ({failures}/{a.max_failures})", flush=True)
            if failures >= a.max_failures:
                raise SystemExit("[sync] remote state unreadable repeatedly — the VM may have been terminated")
            time.sleep(a.interval)
            continue
        # 第 1 项：checkpoint 一保存就下载
        rs, ls = state["last_step"], local_step()
        if rs is not None and (ls is None or rs > ls):
            print(f"[sync] {time.strftime('%H:%M')} new checkpoint step {rs} (local {ls}) -> fetch", flush=True)
            cmd_fetch(argparse.Namespace(session=a.session, run=a.run, drive=a.drive, what="last"))
        if state["done"] and state["best"] and not best_done:
            print(f"[sync] {time.strftime('%H:%M')} training done -> fetch best", flush=True)
            cmd_fetch(argparse.Namespace(session=a.session, run=a.run, drive=a.drive, what="best"))
            best_done = True
        # 第 2 项：评测结果逐个下载（大小两次查询不变才算写完）
        for path, size in state["files"].items():
            if got.get(path) == size:
                continue
            if sizes.get(path) == size:
                local = os.path.join(ROOT, path)
                os.makedirs(os.path.dirname(local), exist_ok=True)
                colab("download", "-s", a.session, "/content/AJev/" + path, local, check=False, timeout=900)
                if os.path.exists(local) and os.path.getsize(local) == size:
                    got[path] = size
                    print(f"[sync] {time.strftime('%H:%M')} downloaded {path} ({size} bytes)", flush=True)
            sizes[path] = size
        pending = [p for p, sz in state["files"].items() if got.get(p) != sz]
        if a.done_file and state["done_file"] and not pending:
            print("[sync] done file present and all files downloaded", flush=True)
            return
        time.sleep(a.interval)


def cmd_stop(a) -> None:
    """``stop`` 子命令：释放 VM。空闲的 VM 也会消耗额度，用完一定要停掉。"""
    colab("stop", "-s", a.session)


def main() -> None:
    """命令行入口：定义各子命令及其参数，然后分派给对应的 ``cmd_*`` 函数。

    “子命令”就像 git 的 ``git commit`` / ``git push``：同一个脚本，第一个参数决定做什么。
    argparse 的 ``add_subparsers`` 用来实现这种结构；``set_defaults(fn=...)`` 把处理函数
    绑在子命令上，解析完参数后统一用 ``a.fn(a)`` 调用。
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        """注册一个子命令：所有子命令都带 ``-s/--session``（默认会话名 ajev），并绑定处理函数。"""
        p = sub.add_parser(name, **kw)
        p.add_argument("-s", "--session", default="ajev")
        p.set_defaults(fn=fn)
        return p

    p = add("up", cmd_up)
    p.add_argument("--gpu", default="T4")
    p.add_argument("--data", help="local data dir to upload (e.g. data/build)")
    add("drive", cmd_drive)
    # train / status / fetch 都需要指定运行名，并可选择运行目录是否放在 Drive 上。
    for name, fn in (("train", cmd_train), ("status", cmd_status), ("fetch", cmd_fetch)):
        p = add(name, fn)
        p.add_argument("--run", required=True)
        p.add_argument("--drive", action="store_true", help=f"run dir under {DRIVE_RUNS}")
    sub.choices["train"].add_argument("--data-dir", default="data/build")
    # REMAINDER：把剩下的所有参数原样收集起来，转发给 ajev.train.train。
    sub.choices["train"].add_argument("train_args", nargs=argparse.REMAINDER,
                                      help="extra args for ajev.train.train (after --)")
    sub.choices["status"].add_argument("--lines", type=int, default=15)
    sub.choices["fetch"].add_argument("--what", default="best")
    p = add("sync", cmd_sync, help="keep downloading new checkpoints / result files as soon as they appear")
    p.add_argument("--run", required=True)
    p.add_argument("--drive", action="store_true")
    p.add_argument("--files", nargs="*", default=[], help="result files to download, globs relative to /content/AJev")
    p.add_argument("--done-file", default="", help="exit once this remote file exists and all files are downloaded")
    p.add_argument("--interval", type=int, default=90)
    p.add_argument("--max-failures", type=int, default=5)
    add("stop", cmd_stop)

    a = ap.parse_args()
    # REMAINDER 会把分隔用的 "--" 一起收进来，这里去掉它。
    if getattr(a, "train_args", None) and a.train_args[0] == "--":
        a.train_args = a.train_args[1:]
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
