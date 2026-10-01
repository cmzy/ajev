#!/usr/bin/env python3
"""Drive training on a Colab VM through the `colab` CLI.

    python scripts/colab_job.py up      -s ajev --gpu T4 --data data/build   # create VM, push code + data, install
    python scripts/colab_job.py drive   -s ajev                              # mount Google Drive at /content/drive
    python scripts/colab_job.py train   -s ajev --run sft1 -- --epochs 3 --grad-ckpt
    python scripts/colab_job.py status  -s ajev --run sft1                   # GPU + last log lines
    python scripts/colab_job.py fetch   -s ajev --run sft1 --what best       # download runs/<run>/<what>
    python scripts/colab_job.py stop    -s ajev

Training runs detached (nohup) on the VM, so the local terminal can disconnect. With
`--drive`, run directories live on Google Drive and survive a reclaimed VM; re-running
`train` with the same --run resumes from the last checkpoint.

Note: code is sent to the VM as a file (`colab exec -f`); piping code on stdin hangs.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = "/content/AJev"
DRIVE_RUNS = "/content/drive/MyDrive/ajev/runs"
CODE_PATHS = ["ajev", "scripts", "pyproject.toml", "README.md"]


def colab(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print("$ colab " + " ".join(shlex.quote(a) for a in args), flush=True)
    return subprocess.run(["colab", *args], check=check)


def remote_python(session: str, code: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        path = f.name
    try:
        colab("exec", "-s", session, "-f", path)
    finally:
        os.unlink(path)


def make_tar(paths: list[str], base: str, name: str) -> str:
    out = os.path.join(tempfile.gettempdir(), name)
    with tarfile.open(out, "w:gz") as tar:
        for p in paths:
            tar.add(os.path.join(base, p), arcname=p,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return out


def runs_dir(drive: bool) -> str:
    return DRIVE_RUNS if drive else f"{REMOTE}/runs"


def cmd_up(a) -> None:
    sessions = subprocess.run(["colab", "sessions"], capture_output=True, text=True).stdout
    if f"[{a.session}]" not in sessions:
        colab("new", "-s", a.session, *(["--gpu", a.gpu] if a.gpu else []))
    code = make_tar(CODE_PATHS, ROOT, "ajev_code.tar.gz")
    colab("upload", "-s", a.session, code, "/content/ajev_code.tar.gz")
    if a.data:
        data = make_tar([os.path.relpath(a.data, ROOT)], ROOT, "ajev_data.tar.gz")
        colab("upload", "-s", a.session, data, "/content/ajev_data.tar.gz")
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
    colab("drivemount", "-s", a.session)
    remote_python(a.session, f"""
        import os
        os.makedirs("{DRIVE_RUNS}", exist_ok=True)
        print("drive runs dir:", "{DRIVE_RUNS}", os.listdir("{DRIVE_RUNS}"))
    """)


def cmd_train(a) -> None:
    out = f"{runs_dir(a.drive)}/{a.run}"
    data = f"{REMOTE}/{a.data_dir}"
    args = ["--train", f"{data}/train.jsonl", "--val", f"{data}/val.jsonl", f"{data}/val_typed.jsonl",
            "--out", out, *a.train_args]
    cmd = " ".join(shlex.quote(x) for x in ["python", "-m", "ajev.train.train", *args])
    remote_python(a.session, f"""
        import os, subprocess
        os.makedirs({out!r}, exist_ok=True)
        log = open({out!r} + "/stdout.log", "a")
        p = subprocess.Popen({cmd!r}, shell=True, cwd="{REMOTE}", stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        open({out!r} + "/pid", "w").write(str(p.pid))
        print("started pid", p.pid, "->", {out!r})
    """)


def cmd_status(a) -> None:
    out = f"{runs_dir(a.drive)}/{a.run}"
    remote_python(a.session, f"""
        import os, subprocess
        print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
        pid = open({out!r} + "/pid").read().strip() if os.path.exists({out!r} + "/pid") else None
        alive = pid is not None and os.path.exists(f"/proc/{{pid}}")
        print("pid", pid, "running" if alive else "NOT running")
        lines = open({out!r} + "/stdout.log").read().splitlines() if os.path.exists({out!r} + "/stdout.log") else []
        print("\\n".join(l[:300] for l in lines[-{a.lines}:]))
    """)


def cmd_fetch(a) -> None:
    src = f"{runs_dir(a.drive)}/{a.run}"
    tar_remote = f"/content/fetch_{a.run}_{a.what}.tar.gz"
    remote_python(a.session, f"""
        import tarfile
        with tarfile.open({tar_remote!r}, "w:gz") as t:
            t.add({src!r} + "/" + {a.what!r}, arcname={a.what!r})
            for f in ("log.jsonl", "args.json"):
                try:
                    t.add({src!r} + "/" + f, arcname=f)
                except FileNotFoundError:
                    pass
    """)
    local_dir = os.path.join(ROOT, "runs", a.run)
    os.makedirs(local_dir, exist_ok=True)
    local_tar = os.path.join(local_dir, f"{a.what}.tar.gz")
    colab("download", "-s", a.session, tar_remote, local_tar)
    with tarfile.open(local_tar) as t:
        t.extractall(local_dir)
    os.remove(local_tar)
    print("fetched ->", local_dir)


def cmd_stop(a) -> None:
    colab("stop", "-s", a.session)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("-s", "--session", default="ajev")
        p.set_defaults(fn=fn)
        return p

    p = add("up", cmd_up)
    p.add_argument("--gpu", default="T4")
    p.add_argument("--data", help="local data dir to upload (e.g. data/build)")
    add("drive", cmd_drive)
    for name, fn in (("train", cmd_train), ("status", cmd_status), ("fetch", cmd_fetch)):
        p = add(name, fn)
        p.add_argument("--run", required=True)
        p.add_argument("--drive", action="store_true", help=f"run dir under {DRIVE_RUNS}")
    sub.choices["train"].add_argument("--data-dir", default="data/build")
    sub.choices["train"].add_argument("train_args", nargs=argparse.REMAINDER,
                                      help="extra args for ajev.train.train (after --)")
    sub.choices["status"].add_argument("--lines", type=int, default=15)
    sub.choices["fetch"].add_argument("--what", default="best")
    add("stop", cmd_stop)

    a = ap.parse_args()
    if getattr(a, "train_args", None) and a.train_args[0] == "--":
        a.train_args = a.train_args[1:]
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
