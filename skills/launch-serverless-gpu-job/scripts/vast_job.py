#!/usr/bin/env python3
"""vast_job.py - Vast.ai GPU job lifecycle helper for the launch-serverless-gpu-job skill.

One helper so renting a GPU, driving a job on it, and tearing it down is a few lines.
Encodes the edges found running real GPU training jobs on Vast.ai's marketplace:
2FA-gated instance ops, transient offers, `-y` on destroy, `instances-v1` polling, the
`</dev/null` ssh-detach trap, and banner-tolerant ssh.

Commands:
  preflight                          auth + 2FA-session + ssh-key check
  tfa-login  --code NNNNNN           open a ~7-day 2FA session (required for instance ops)
  launch     [filters] [--onstart F] search cheapest matching offer(s), create, retry churn
  wait  <id> [--timeout S]           poll until running; print COLD_START
  list                               instances (+ $/hr, ssh endpoint)
  cost                               credit + total running $/hr + runway estimate
  audit                              hygiene sweep: EVERYTHING that still bills, any state
                                     (running bills GPU+disk; stopped still bills disk);
                                     exit 0 only when the account is clean
  ssh   <id> --cmd "..."             run a command on the box (robust; banners stripped)
  exec  <id> --cmd "..." [--log P]   start a DETACHED process (survives disconnect)
  tail  <id> <path> [-n N]           tail a remote log
  pull  <id> <remote> <local> [-r]   scp a file/dir down (e.g. checkpoints)
  logs  <id>                         the onstart log (/var/log/onstart.log)
  guard <id> --max-hours H [--done-file P]   auto-destroy on JOB_DONE marker or timeout
  destroy <id>                       destroy (always -y; else the CLI hangs while billing)
  destroy-all                        destroy every running instance
"""
import argparse, json, os, shlex, subprocess, sys, time

HOME = os.path.expanduser("~")


def _default_vast_bin():
    """Find the vastai CLI across OSes; fall back to PATH."""
    for cand in (
        os.path.join(HOME, ".vast-cli", "Scripts", "vastai.exe"),  # Windows venv
        os.path.join(HOME, ".vast-cli", "bin", "vastai"),          # Linux/macOS venv
    ):
        if os.path.exists(cand):
            return cand
    return "vastai"  # rely on PATH (e.g. `pipx install vastai`)


VAST = os.environ.get("VAST_BIN", _default_vast_bin())
STATE = os.path.join(HOME, ".vast-cli", "state.json")
PUBKEY = os.environ.get("VAST_PUBKEY", os.path.join(HOME, ".ssh", "id_ed25519.pub"))
os.environ.setdefault("PYTHONUTF8", "1")
DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"

_BANNER = ("Welcome to vast.ai", "Have fun!", "authentication fails",
           "Permanently added", "double check your ssh key")
_CONN_FAIL = ("Permission denied", "Connection refused", "Connection timed out",
              "Could not resolve", "kex_exchange", "Connection closed",
              "Operation timed out", "No route to host", "Connection reset")
_ATTACHED = set()


def vast(*args, timeout=180):
    r = subprocess.run([VAST, *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout or "", r.stderr or ""


def parse_json_blob(*texts):
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        try:
            return json.loads(t)
        except Exception:
            for c in ("{", "["):
                i = t.find(c)
                if i >= 0:
                    try:
                        return json.loads(t[i:])
                    except Exception:
                        pass
    return None


def is_2fa_error(*texts):
    b = " ".join(texts)
    return "Two Factor Authentication" in b or "requires a 2FA session" in b


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def save_state(s):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(s, open(STATE, "w"), indent=2)


def user_credit():
    _, out, _ = vast("show", "user", "--raw")
    u = parse_json_blob(out)
    if not u:
        return None
    return (u[0] if isinstance(u, list) else u).get("credit")


def list_instances():
    _, out, err = vast("show", "instances-v1", "--raw")
    if is_2fa_error(out, err):
        return None, "NEED_TFA"
    d = parse_json_blob(out)
    if isinstance(d, dict):
        d = d.get("instances") or d.get("results") or []
    return (d or []), ""


def find_instance(iid):
    d, err = list_instances()
    if d is None:
        return None, err
    for i in d:
        if str(i.get("id")) == str(iid):
            return i, ""
    return None, ""


# ---- robust SSH -----------------------------------------------------------
def _strip(text):
    return "\n".join(l for l in (text or "").splitlines()
                     if not any(b in l for b in _BANNER)).strip()


def _sshbase(inst):
    return ["ssh", "-p", str(inst["ssh_port"]),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=15",
            f"root@{inst['ssh_host']}"]


def attach_key(inst):
    iid = str(inst["id"])
    if iid in _ATTACHED or not os.path.exists(PUBKEY):
        return
    vast("attach", "ssh", iid, open(PUBKEY).read().strip(), timeout=60)
    _ATTACHED.add(iid)


def ssh_run(inst, remote_cmd, timeout=120, tries=6):
    """Run a command over ssh. Retries ONLY on connection-level failures - NOT on a
    nonzero exit from the remote command (that is a valid result). Vast's login banner
    is stripped. Returns (stdout, stderr, returncode)."""
    attach_key(inst)
    base = _sshbase(inst)
    last = ""
    for k in range(tries):
        try:
            r = subprocess.run(base + [remote_cmd], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = "ssh timeout"; time.sleep(6); continue
        conn_fail = r.returncode == 255 or any(m in (r.stderr or "") for m in _CONN_FAIL)
        if conn_fail and k < tries - 1:
            last = (r.stderr or "").strip(); time.sleep(6); continue
        return _strip(r.stdout), _strip(r.stderr), r.returncode
    return "", last, 255


def _resolve(iid):
    inst, err = find_instance(iid)
    if inst is None:
        print("NEED_TFA - run tfa-login" if err == "NEED_TFA" else f"instance {iid} not found")
        sys.exit(3 if err == "NEED_TFA" else 1)
    return inst


# ---- commands -------------------------------------------------------------
def cmd_preflight(a):
    import shutil
    if not (os.path.exists(VAST) or shutil.which(VAST)):
        print(f"FAIL: vastai not found at {VAST} (set VAST_BIN or add vastai to PATH)"); return 2
    cr = user_credit()
    if cr is None:
        print("FAIL: `show user` returned no JSON - is the API key set? (vastai set api-key <KEY>)"); return 2
    print(f"OK: authenticated, credit=${cr:.2f}")
    d, err = list_instances()
    if d is None:
        print("FAIL: instance ops BLOCKED (no 2FA session). Run:\n"
              "  python vast_job.py tfa-login --code <6-digit TOTP>   (session lasts ~7 days)")
        return 3
    print(f"OK: 2FA session active. Running instances: {len(d)}")
    print("OK: SSH key present." if os.path.exists(PUBKEY)
          else f"WARN: no SSH key at {PUBKEY} - ssh/exec/tail/pull will not work.")
    return 0


def cmd_tfa_login(a):
    _, out, err = vast("tfa", "login", "--method-type", "totp", "-c", str(a.code))
    print((out or err).strip())
    return 0 if "successful" in (out + err).lower() else 1


def cmd_launch(a):
    q = (f"rentable=true verified=true num_gpus={a.num_gpus} direct_port_count>=1 "
         f"disk_space>={a.disk} inet_down>={a.min_down}")
    if a.gpu:
        q += f" gpu_name={a.gpu}"
    if a.gpu_ram:
        q += f" gpu_ram>={a.gpu_ram}"
    _, out, err = vast("search", "offers", q, "-o", "dph+", "--raw")
    if is_2fa_error(out, err):
        print("NEED_TFA - run tfa-login first"); return 3
    offers = [o for o in (parse_json_blob(out) or []) if o.get("dph_total", 1e9) <= a.max_price]
    offers.sort(key=lambda o: o["dph_total"])
    if not offers:
        print(f"FAIL: no offers (gpu={a.gpu or 'any'} max_price={a.max_price} min_down={a.min_down} disk>={a.disk})")
        return 1
    print(f"{len(offers)} candidate offers; trying cheapest {min(a.tries, len(offers))}...")
    for o in offers[:a.tries]:
        oid = str(o["id"])
        args = ["create", "instance", oid, "--image", a.image, "--disk", str(a.disk),
                "--ssh", "--direct", "--cancel-unavail", "--label", a.label]
        if a.onstart:
            args += ["--onstart", a.onstart]
        args += ["--raw"]
        t0 = time.time()
        _, out, err = vast(*args)
        resp = parse_json_blob(out, err) or {}
        if resp.get("success") and resp.get("new_contract"):
            iid = resp["new_contract"]
            st = load_state()
            st[str(iid)] = {"create_ts": t0, "gpu": o.get("gpu_name"), "dph": o.get("dph_total"),
                            "label": a.label}
            save_state(st)
            print(f"LAUNCHED instance_id={iid} gpu={o.get('gpu_name')} "
                  f"${o.get('dph_total'):.3f}/hr down={o.get('inet_down')}Mbps")
            if a.max_hours:
                print(f"NOTE: run `guard {iid} --max-hours {a.max_hours}` (in the background) to auto-destroy.")
            return 0
        msg = (resp.get("msg") or err or out).strip()[:120]
        if "no_such_ask" in msg or "not available" in msg:
            print(f"  offer {oid} unavailable, next..."); continue
        print(f"FAIL: create error on {oid}: {msg}"); return 1
    print(f"FAIL: all {a.tries} candidates unavailable (marketplace churn); retry."); return 1


def cmd_wait(a):
    ct = load_state().get(str(a.id), {}).get("create_ts")
    deadline = time.time() + a.timeout
    last = None
    while time.time() < deadline:
        inst, err = find_instance(a.id)
        if inst is None and err == "NEED_TFA":
            print("NEED_TFA"); return 3
        st = inst.get("actual_status") if inst else None
        if st != last:
            el = (time.time() - ct) if ct else -1
            print(f"[{el:5.0f}s] status={st}"); last = st
        if st == "running":
            if ct:
                print(f">>> COLD_START(create->running) = {time.time()-ct:.0f}s")
            return 0
        time.sleep(6)
    print("TIMEOUT: not running"); return 1


def cmd_list(a):
    d, err = list_instances()
    if d is None:
        print("NEED_TFA"); return 3
    if not d:
        print("no instances"); return 0
    for i in d:
        print(f"  id={i.get('id')} {i.get('actual_status')} {i.get('gpu_name')} "
              f"${i.get('dph_total',0):.3f}/hr {i.get('ssh_host')}:{i.get('ssh_port')} label={i.get('label')}")
    return 0


def cmd_cost(a):
    cr = user_credit()
    d, err = list_instances()
    if d is None:
        print("NEED_TFA"); return 3
    total = sum(i.get("dph_total", 0) for i in d)
    print(f"credit: ${cr:.2f}" if cr is not None else "credit: ?")
    for i in d:
        print(f"  id={i.get('id')} {i.get('gpu_name')} ${i.get('dph_total',0):.3f}/hr")
    line = f"running total: ${total:.3f}/hr"
    if cr and total:
        line += f"  (~{cr/total:.1f}h GPU-only runway)"
    print(line)
    print("NOTE: Vast also bills disk (~$0.10-0.15/GB/mo) + one-time inbound bandwidth on top.")
    return 0


def cmd_audit(a):
    """Hygiene sweep: list EVERY instance in ANY state and what it is still billing.
    A running instance bills GPU + disk; a STOPPED/exited one still bills its disk
    (~$0.10-0.15/GB/mo) until destroyed. Exit 0 = account clean; exit 1 = something
    is billing (so scripts/agents can gate on it)."""
    cr = user_credit()
    d, err = list_instances()
    if d is None:
        print("NEED_TFA"); return 3
    if not d:
        print("CLEAN: no instances in any state - nothing billing GPU or disk.")
        if cr is not None:
            print(f"credit: ${cr:.2f}")
        return 0
    now = time.time()
    gpu_total, disk_total = 0.0, 0.0
    for i in d:
        st = (i.get("actual_status") or i.get("cur_state") or "?").lower()
        dph = i.get("dph_total") or 0
        disk = i.get("disk_space") or 0
        age = f"{(now - i['start_date'])/3600:.1f}h" if i.get("start_date") else "?"
        disk_total += disk
        if st == "running":
            gpu_total += dph
            note = f"RUNNING: bills ${dph:.3f}/hr GPU + disk"
        else:
            note = f"{st.upper()}: GPU off but its {disk:.0f}GB DISK STILL BILLS until destroyed"
        print(f"  id={i.get('id')} {i.get('gpu_name')} label={i.get('label')} "
              f"disk={disk:.0f}GB age={age} -> {note}")
    print(f"TOTAL: ${gpu_total:.3f}/hr GPU + ~${disk_total*0.10:.2f}-{disk_total*0.15:.2f}/mo disk "
          f"({disk_total:.0f} GB allocated)")
    if cr is not None:
        line = f"credit: ${cr:.2f}"
        if gpu_total:
            line += f"  (~{cr/gpu_total:.1f}h GPU-only runway)"
        print(line)
    print("Anything above is spending money. `destroy <id>` what you don't need - a stopped")
    print("instance is NOT free. Deeper history: `vastai show invoices-v1` / `show audit-logs`.")
    return 1


def destroy_verified(iid):
    """Destroy an instance and CONFIRM it's gone. Returns an exit code:
    0 = confirmed destroyed, 3 = 2FA gone so it could NOT be actioned/verified
    (may still bill), 1 = command ran but the box is still present. Money safety
    depends on never reporting success we didn't verify."""
    rc, out, err = vast("destroy", "instance", str(iid), "-y", timeout=60)
    if is_2fa_error(out, err):
        print(f"FAIL: destroy BLOCKED for {iid} (no 2FA session) - IT MAY STILL BE BILLING. "
              f"Run `tfa-login --code <TOTP>` then `destroy {iid}`.")
        return 3
    st = load_state(); st.pop(str(iid), None); save_state(st)
    inst, ferr = find_instance(iid)
    if ferr == "NEED_TFA":
        print(f"WARN: destroy sent for {iid} but can't verify (2FA session gone). "
              f"Re-run `tfa-login` then `list` to confirm it's gone.")
        return 3
    if inst is None:
        print(f"destroyed {iid}"); return 0
    print(f"WARN: {iid} STILL PRESENT ({inst.get('actual_status')}) - retry `destroy {iid}`."); return 1


def cmd_ssh(a):
    inst = _resolve(a.id)
    out, err, rc = ssh_run(inst, a.cmd or "hostname && nvidia-smi -L", timeout=a.timeout)
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    return rc  # propagate the remote command's status so callers can gate on it


def cmd_exec(a):
    inst = _resolve(a.id)
    log = a.log or "/root/job.log"
    # </dev/null is REQUIRED or the ssh channel hangs on the backgrounded process.
    cmd = f"nohup bash -lc {shlex.quote(a.cmd)} </dev/null >{shlex.quote(log)} 2>&1 & echo EXEC_PID $!"
    out, err, rc = ssh_run(inst, cmd, timeout=90)
    if rc != 0 or "EXEC_PID" not in out:
        # No background process was started - don't let automation tail a log that never appears.
        print(f"FAIL: could not start detached job (ssh rc={rc}): {(err or out).strip()[:160]}")
        return 1
    print(out.strip(), f"(log: {log}; watch with `tail {a.id} {log}`)")
    return 0


def cmd_tail(a):
    inst = _resolve(a.id)
    out, err, rc = ssh_run(inst, f"tail -n {a.n} {shlex.quote(a.path)} | tr -d '\\r'", timeout=a.timeout)
    print(out or err or "(empty)")
    return 0


def cmd_pull(a):
    inst = _resolve(a.id)
    attach_key(inst)
    if not os.path.isdir(a.local):
        os.makedirs(os.path.dirname(os.path.abspath(a.local)), exist_ok=True)
    base = ["scp", "-P", str(inst["ssh_port"]), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=30"]
    if a.recursive:
        base.append("-r")
    src = f"root@{inst['ssh_host']}:{a.remote}"
    for k in range(4):
        r = subprocess.run(base + [src, a.local], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"pulled {a.remote} -> {a.local}"); return 0
        if any(m in (r.stderr or "") for m in _CONN_FAIL) and k < 3:
            time.sleep(6); continue
        print(f"FAIL: {(r.stderr or '').strip()[:160]}"); return 1
    return 1


def cmd_logs(a):
    rc, out, err = vast("logs", str(a.id), "--tail", "200")
    if rc == 0 and "401" not in out and "lacks" not in (out + err):
        print(out); return 0
    print("(`vastai logs` unavailable on this key - reading /var/log/onstart.log over SSH)")
    inst = _resolve(a.id)
    out, err, rc = ssh_run(inst, "tail -n 200 /var/log/onstart.log 2>/dev/null | tr -d '\\r'")
    print(out or "(empty)")
    return 0


def cmd_guard(a):
    """Local watchdog: destroy the instance when <done-file> appears or after --max-hours.
    Best-effort money safety. NOTE: it needs THIS process alive (run it in the background);
    it is not a true instance-side self-destruct."""
    deadline = time.time() + a.max_hours * 3600
    print(f"guarding {a.id}: destroy on {a.done_file} or after {a.max_hours}h", flush=True)
    while time.time() < deadline:
        inst, err = find_instance(a.id)
        if inst is None:
            if err == "NEED_TFA":
                # Can't observe the box, so we can't guarantee teardown - fail loud, don't assume gone.
                print(f"WARN: 2FA session expired - guard can no longer observe or destroy {a.id}; "
                      f"IT MAY STILL BE BILLING. Run `tfa-login` then `destroy {a.id}`.", flush=True)
                return 3
            print("instance already gone"); return 0
        out, _, _ = ssh_run(inst, f"test -f {shlex.quote(a.done_file)} && echo DONE || echo RUN", tries=3)
        if "DONE" in out:
            print("JOB_DONE seen -> destroying", flush=True)
            return destroy_verified(a.id)
        time.sleep(a.interval)
    print(f"max-hours reached -> destroying {a.id}", flush=True)
    return destroy_verified(a.id)


def cmd_destroy(a):
    return destroy_verified(a.id)


def cmd_destroy_all(a):
    d, err = list_instances()
    if d is None:
        print("NEED_TFA"); return 3
    failed = [i["id"] for i in d if destroy_verified(i["id"]) != 0]
    save_state({})
    if failed:
        print(f"WARN: {len(failed)} instance(s) NOT confirmed destroyed: {failed} - run `audit`."); return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight").set_defaults(fn=cmd_preflight)
    q = sub.add_parser("tfa-login"); q.add_argument("--code", required=True); q.set_defaults(fn=cmd_tfa_login)

    l = sub.add_parser("launch")
    l.add_argument("--gpu", default=""); l.add_argument("--gpu-ram", type=int, default=0)
    l.add_argument("--num-gpus", type=int, default=1); l.add_argument("--max-price", type=float, default=0.6)
    l.add_argument("--min-down", type=int, default=2000); l.add_argument("--disk", type=int, default=40)
    l.add_argument("--image", default=DEFAULT_IMAGE); l.add_argument("--onstart", default="")
    l.add_argument("--label", default="vast-job"); l.add_argument("--tries", type=int, default=8)
    l.add_argument("--max-hours", type=float, default=0); l.set_defaults(fn=cmd_launch)

    w = sub.add_parser("wait"); w.add_argument("id"); w.add_argument("--timeout", type=int, default=600); w.set_defaults(fn=cmd_wait)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("cost").set_defaults(fn=cmd_cost)
    sub.add_parser("audit").set_defaults(fn=cmd_audit)
    s = sub.add_parser("ssh"); s.add_argument("id"); s.add_argument("--cmd", default=""); s.add_argument("--timeout", type=int, default=120); s.set_defaults(fn=cmd_ssh)
    e = sub.add_parser("exec"); e.add_argument("id"); e.add_argument("--cmd", required=True); e.add_argument("--log", default=""); e.set_defaults(fn=cmd_exec)
    t = sub.add_parser("tail"); t.add_argument("id"); t.add_argument("path"); t.add_argument("-n", type=int, default=40); t.add_argument("--timeout", type=int, default=60); t.set_defaults(fn=cmd_tail)
    pl = sub.add_parser("pull"); pl.add_argument("id"); pl.add_argument("remote"); pl.add_argument("local"); pl.add_argument("-r", "--recursive", action="store_true"); pl.set_defaults(fn=cmd_pull)
    g = sub.add_parser("logs"); g.add_argument("id"); g.set_defaults(fn=cmd_logs)
    gu = sub.add_parser("guard"); gu.add_argument("id"); gu.add_argument("--max-hours", type=float, required=True)
    gu.add_argument("--done-file", dest="done_file", default="/root/JOB_DONE"); gu.add_argument("--interval", type=int, default=300); gu.set_defaults(fn=cmd_guard)
    d = sub.add_parser("destroy"); d.add_argument("id"); d.set_defaults(fn=cmd_destroy)
    sub.add_parser("destroy-all").set_defaults(fn=cmd_destroy_all)

    a = p.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
