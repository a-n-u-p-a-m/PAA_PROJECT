#!/usr/bin/env python3
"""Does this machine support the measurements our project needs?

Run this FIRST on any new GPU -- a Colab runtime, a Kaggle session, a
metered workstation slot --
before trusting a single energy number from it. Each check maps to a concrete
project phase, and the summary says which phases are safe to run here.

    python probe_env.py              # full probe, ~2 minutes
    python probe_env.py --quick      # skip the stability sweep
    python probe_env.py --json out.json

The one that matters most is the LAST check. CLEAR validates a measurement by
requiring the relative standard deviation across trials to stay under 10%.
On a dedicated GPU with locked clocks that is easy; on a shared, thermally
throttled, frequency-boosting cloud GPU it very often is not -- and a probe
that fails there tells you up front, instead of after a week of sweeps.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""

OK, WARN, FAIL = f"{GREEN}PASS{RESET}", f"{YELLOW}WARN{RESET}", f"{RED}FAIL{RESET}"

results: dict[str, Any] = {}


def say(status: str, label: str, detail: str = "") -> None:
    print(f"  [{status}] {label}" + (f"\n         {DIM}{detail}{RESET}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# ---------------------------------------------------------------- host ----


def detect_environment() -> tuple[str, str]:
    """Identify the host, and say which signal gave it away.

    Order and specificity both matter: a Kaggle image also contains
    ``/content``, so a Colab check that looks for that directory will claim a
    Kaggle session. Environment variables set by each platform's own runtime
    are unambiguous, so they are checked first and directories only used as a
    last resort.
    """
    kaggle_vars = ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",
                   "KAGGLE_CONTAINER_NAME", "KAGGLE_DATA_PROXY_TOKEN")
    colab_vars = ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_JUPYTER_IP")

    for var in kaggle_vars:
        if os.environ.get(var):
            run = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")
            return f"Kaggle{f' ({run})' if run else ''}", f"${var}"
    for var in colab_vars:
        if os.environ.get(var):
            return "Google Colab", f"${var}"
    try:
        import google.colab  # noqa: F401

        return "Google Colab", "google.colab importable"
    except ImportError:
        pass

    # directory fallbacks, least reliable
    if os.path.isdir("/kaggle/input") or os.path.isdir("/kaggle/working"):
        return "Kaggle", "/kaggle/* present"
    if os.path.isdir("/content/drive") or os.path.isdir("/content/sample_data"):
        return "Google Colab", "/content/* present"
    if os.path.exists("/.dockerenv"):
        return "container", "/.dockerenv"
    return "bare metal / cluster node", "no platform markers"


def probe_host() -> None:
    section("Host")
    env, signal_name = detect_environment()
    results["environment"] = env
    results["environment_signal"] = signal_name
    results["python"] = platform.python_version()
    say(OK, f"Environment: {env}",
        f"detected via {signal_name}; python {platform.python_version()} "
        f"on {platform.platform()}")

    try:
        import torch

        results["torch"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()
        say(
            OK if torch.cuda.is_available() else FAIL,
            f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}",
        )
    except ImportError:
        say(FAIL, "torch not installed", "pip install torch")
        sys.exit(1)


# ----------------------------------------------------------------- gpu ----


def probe_gpu() -> bool:
    section("GPU")
    import torch

    if not torch.cuda.is_available():
        say(FAIL, "No CUDA device", "Every energy measurement in this project needs one.")
        return False

    n_gpu = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    cc = f"{props.major}.{props.minor}"
    total_gb = props.total_memory / 1024**3
    results.update(
        gpu_name=props.name,
        gpu_count=n_gpu,
        compute_capability=cc,
        vram_total_gb=round(total_gb, 2),
        sm_count=props.multi_processor_count,
    )
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        say(OK, f"GPU {i}: {p.name}  (CC {p.major}.{p.minor}, "
                f"{p.total_memory / 1024**3:.1f} GB, "
                f"{p.multi_processor_count} SMs)")
    if n_gpu > 1:
        say(OK, f"{n_gpu} GPUs available",
            "Each sweep is single-GPU and batch 1, so the second card does "
            "not speed one run up -- but two independent runs (say NF4 and "
            "fp16) can go in parallel with CUDA_VISIBLE_DEVICES=0 and =1, "
            "halving wall time. Checks below describe GPU 0.")

    # A GPU newer than the installed PyTorch build has no compiled kernels,
    # and the failure is a runtime "no kernel image is available" on the first
    # matmul -- after the model has loaded. On a metered slot that is an
    # expensive way to find out, so check it before anything else.
    try:
        arch_list = torch.cuda.get_arch_list()
        sm = f"sm_{props.major}{props.minor}"
        results["torch_arch_list"] = arch_list
        if arch_list and sm not in arch_list:
            say(
                FAIL,
                f"This torch build has no {sm} kernels",
                f"Built for: {', '.join(arch_list)}. A Blackwell card (sm_120) "
                "needs torch >= 2.7 with the cu128 wheels; the 2.6 build the "
                "paper used predates it. Reinstall with\n         "
                "pip install torch --index-url "
                "https://download.pytorch.org/whl/cu128",
            )
        else:
            say(OK, f"torch has {sm} kernels")
    except Exception:
        pass

    if props.major >= 10:
        say(OK, "Blackwell or newer: native bf16 and fp8 available",
            "Keep bnb_4bit_compute_dtype=float16 if results from an older card "
            "share a table with these -- pre-Ampere GPUs have no bf16, and the "
            "dtype must match across devices for the comparison to mean anything.")

    # the hard architectural gate on NVML's energy counter
    if props.major < 7:
        say(
            FAIL,
            f"Compute capability {cc} is pre-Volta",
            "nvmlDeviceGetTotalEnergyConsumption needs CC >= 7.0. A Kaggle P100 "
            "(CC 6.0) lands here: energy profiling is impossible on this device, "
            "though the decoding pipeline itself will run fine.",
        )
    else:
        say(OK, f"CC {cc} >= 7.0, so the NVML energy counter should exist")

    free = torch.cuda.mem_get_info()[0] / 1024**3
    results["vram_free_gb"] = round(free, 2)
    say(OK, f"{free:.1f} GB free right now")
    return True


def probe_model_fit() -> None:
    """Which of the paper's eight pairs can this device actually hold?"""
    section("Dutta et al. model pairs  (arXiv:2602.09113, Table 3)")
    sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
    from specdec.configs import MODEL_PAIRS, feasibility_table

    import torch

    free = torch.cuda.mem_get_info()[0] / 1024**3
    print(feasibility_table(free))

    fp16_ok = [
        p.key for p in MODEL_PAIRS.values()
        if p.supported and p.fits(free, "float16", "float16")
    ]
    nf4_ok = [
        p.key for p in MODEL_PAIRS.values()
        if p.supported and p.fits(free, "nf4", "float16")
    ]
    results["pairs_fp16"], results["pairs_nf4"] = fp16_ok, nf4_ok

    print()
    if fp16_ok:
        say(OK, f"unquantized (fp16): {', '.join(fp16_ok)}",
            "Preferred for the energy work -- no dequantization kernel in the way.")
    else:
        say(WARN, "no pair fits unquantized", "NF4 only, which changes what you measure.")
    if nf4_ok:
        say(OK, f"NF4, as the paper used: {', '.join(nf4_ok)}")


def probe_quantization() -> None:
    section("NF4 quantization  (the paper's target precision)")
    import torch

    try:
        import bitsandbytes  # noqa: F401

        results["bitsandbytes"] = True
        say(OK, "bitsandbytes present")
    except Exception as exc:
        results["bitsandbytes"] = False
        say(WARN, f"bitsandbytes not importable: {exc}",
            "pip install bitsandbytes. Only needed to reproduce the paper's NF4 setup.")
        return

    major = torch.cuda.get_device_properties(0).major
    # NF4/FP4 needs Pascal or newer; LLM.int8() needs Turing or newer
    say(OK if major >= 6 else FAIL,
        f"NF4 needs Pascal (CC 6.0)+; this device is CC "
        f"{major}.{torch.cuda.get_device_properties(0).minor}")
    if major == 7 and torch.cuda.get_device_properties(0).minor == 0:
        say(WARN, "Volta: no native bf16",
            "Set bnb_4bit_compute_dtype=float16. bfloat16 will be emulated or "
            "rejected. run_baseline.py defaults to float16 for this reason.")
    say(WARN if major >= 6 else FAIL, "NF4 changes what you are measuring",
        "Every quantized matmul carries a dequantization kernel, so a "
        "component-level breakdown of an NF4 model describes quantized "
        "inference. Measure fp16 too wherever VRAM allows, and report both.")


# ---------------------------------------------------------------- nvml ----


def probe_nvml() -> Any:
    section("NVML energy counter  (phases 3-5: all CLEAR measurement)")
    try:
        import pynvml
    except ImportError:
        say(FAIL, "pynvml not installed", "pip install nvidia-ml-py")
        results["nvml"] = False
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as exc:
        say(FAIL, f"nvmlInit failed: {exc}")
        results["nvml"] = False
        return None

    driver = pynvml.nvmlSystemGetDriverVersion()
    driver = driver.decode() if isinstance(driver, bytes) else driver
    results["driver"] = driver
    say(OK, f"NVML up, driver {driver}")

    try:
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        results["energy_counter"] = True
        say(OK, "nvmlDeviceGetTotalEnergyConsumption works", f"reads {e0 / 1000:.1f} J cumulative")
    except Exception as exc:
        results["energy_counter"] = False
        say(
            FAIL,
            f"Energy counter unavailable: {exc}",
            "No CLEAR-style measurement on this machine. Fall back to integrating "
            "nvmlDeviceGetPowerUsage over time -- far coarser, and not good enough "
            "for component-level numbers.",
        )
        return handle

    # does it actually tick? a counter that exists but never moves is worse
    # than one that errors, because it silently produces zeros
    import torch

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    t0 = time.perf_counter()
    for _ in range(60):
        a = a @ a.T if a.shape[0] == a.shape[1] else a
        a = torch.nn.functional.normalize(a.float(), dim=-1).half()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    delta = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - e0
    results["energy_counter_ticks_mj"] = delta
    if delta <= 0:
        say(FAIL, "Energy counter did not move under load", "It is present but not reporting.")
    else:
        say(
            OK,
            f"Counter advanced {delta} mJ over {dt * 1000:.0f} ms of matmul "
            f"({delta / dt / 1000:.0f} W average)",
        )

    try:
        watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        results["power_w"], results["power_limit_w"] = watts, limit
        say(OK, f"Power telemetry: {watts:.1f} W now, {limit:.0f} W enforced limit")
    except Exception as exc:
        say(WARN, f"Power telemetry unavailable: {exc}")

    return handle


# ------------------------------------------------------- clock control ----


def probe_clock_control(handle) -> None:
    section("Clock stability  (the main threat to reproducible energy numbers)")
    if handle is None:
        return
    import pynvml

    try:
        cur = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        mx = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        results["sm_clock_mhz"], results["sm_clock_max_mhz"] = cur, mx
        say(OK, f"SM clock {cur} / {mx} MHz")
    except Exception as exc:
        say(WARN, f"Clock query failed: {exc}")
        mx = None

    # Locking clocks is what removes DVFS variance between trials. It needs
    # root AND a driver that permits it -- on hosted notebooks it usually does not.
    locked = False
    if mx:
        try:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, int(mx * 0.8), int(mx * 0.8))
            locked = True
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        except Exception as exc:
            results["clock_lock_error"] = str(exc)
    results["can_lock_clocks"] = locked
    if locked:
        say(OK, "Can lock SM clocks", "Lock them before every measurement run; it is the single biggest variance reduction available.")
    else:
        say(
            WARN,
            "Cannot lock SM clocks (no permission)",
            "Expected on Colab/Kaggle. Clocks will boost and throttle between "
            "trials, inflating variance. Mitigate with a fixed warm-up, more "
            "trials, and reporting median rather than mean -- but numbers stay "
            "less reproducible than on a node you control.",
        )

    try:
        mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
        results["persistence_mode"] = bool(mode)
        say(OK if mode else WARN, f"Persistence mode: {'on' if mode else 'off'}",
            "" if mode else "Driver may unload between runs, resetting the cumulative energy counter.")
    except Exception as exc:
        say(WARN, f"Persistence mode unknown: {exc}")

    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        others = [p for p in procs if p.pid != os.getpid()]
        results["other_gpu_processes"] = len(others)
        say(
            OK if not others else FAIL,
            f"{len(others)} other process(es) on this GPU",
            "" if not others else "Any co-tenant workload contaminates every energy reading. Do not measure here.",
        )
    except Exception:
        pass


# ------------------------------------------------------------ profilers ---


def probe_profilers() -> None:
    section("Nsight profilers  (optional: kernel-level attribution)")
    for tool, why in (("nsys", "timeline / phase boundaries"), ("ncu", "per-kernel counters")):
        path = shutil.which(tool)
        results[f"{tool}_present"] = bool(path)
        if not path:
            say(WARN, f"{tool} not on PATH", f"Used for {why}. Not required for the CLEAR pipeline.")
            continue
        say(OK, f"{tool} found at {path}")

    # ncu needs GPU performance counters, which most hosted notebooks withhold
    params = "/proc/driver/nvidia/params"
    if os.path.exists(params):
        try:
            with open(params) as fh:
                text = fh.read()
            for line in text.splitlines():
                if "RestrictProfilingToAdminUsers" in line:
                    restricted = line.strip().endswith("1")
                    results["profiling_restricted"] = restricted
                    say(
                        WARN if restricted else OK,
                        f"NVreg_RestrictProfilingToAdminUsers = {int(restricted)}",
                        "Counters are admin-only; ncu will fail with ERR_NVGPUCTRPERM "
                        "unless you are root. Common on hosted notebooks."
                        if restricted
                        else "Performance counters are accessible.",
                    )
        except Exception:
            pass


# ----------------------------------------------- measurement stability ----


def sensor_period_ms(handle, probe_seconds: float = 2.5) -> Optional[float]:
    """How often does the energy counter actually advance?

    The counter has millijoule resolution but is refreshed by a power sampler
    running at its own cadence -- tens to hundreds of milliseconds depending
    on the board. Measure a window shorter than that cadence and most trials
    read zero while a few catch a whole tick, which looks like enormous
    variance but is really just aliasing. Knowing the true period is what
    lets the window be sized correctly, and it is also the number that sets
    the minimum amplification window for CLEAR on this device.
    """
    import pynvml
    import torch

    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    w = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    for _ in range(3000):  # queue work asynchronously; keeps the GPU busy
        torch.mm(x, w)

    intervals: list[float] = []
    last = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    t0 = t_prev = time.perf_counter()
    while time.perf_counter() - t0 < probe_seconds:
        value = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        if value != last:
            now = time.perf_counter()
            intervals.append((now - t_prev) * 1000)
            t_prev, last = now, value
        time.sleep(0.0005)
    torch.cuda.synchronize()

    if len(intervals) < 3:
        return None
    intervals.sort()
    return intervals[len(intervals) // 2]


def probe_stability(handle, trials: int = 15, reps: Optional[int] = None) -> None:
    """The decisive check: can this machine meet CLEAR's <10% std-dev bar?

    Runs an identical fixed workload ``trials`` times and looks at the spread
    of the per-trial energy. The workload is deliberately small and repeated
    ``reps`` times -- the same amplification trick CLEAR uses to lift a
    microsecond-scale component above the sensor's millisecond cadence.
    """
    section("Measurement stability  (CLEAR's acceptance criterion: rel. std dev < 10%)")
    if handle is None or not results.get("energy_counter"):
        say(WARN, "Skipped -- no energy counter")
        return

    import pynvml
    import torch

    period = sensor_period_ms(handle)
    results["sensor_period_ms"] = round(period, 1) if period else None
    if period:
        say(OK, f"Energy counter advances about every {period:.0f} ms",
            "This is the floor on any single measurement window, and the "
            "reason CLEAR amplifies: one component execution is microseconds, "
            "so it must be repeated until the total clears this cadence.")
    else:
        say(WARN, "Could not measure the counter's update period")

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    w = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    def run(n: int) -> None:
        for _ in range(n):
            torch.mm(x, w)

    # Size the window to the sensor, not to a guess. A window shorter than
    # the update period aliases: most trials read zero, a few catch a whole
    # tick, and the result looks like 200% instability when the GPU is fine.
    if reps is None:
        run(50)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(200)
        torch.cuda.synchronize()
        per_rep_ms = (time.perf_counter() - t0) * 1000 / 200
        target_ms = max(10 * (period or 50.0), 300.0)
        reps = max(200, int(target_ms / max(per_rep_ms, 1e-6)))
        say(OK, f"Auto-sized to {reps} reps per trial "
                f"(~{reps * per_rep_ms:.0f} ms, about "
                f"{reps * per_rep_ms / (period or 50):.0f}x the sensor period)")

    def amplified() -> None:
        run(reps)

    for _ in range(3):  # warm up clocks and caches
        amplified()
    torch.cuda.synchronize()

    energies, times = [], []
    for _ in range(trials):
        torch.cuda.synchronize()
        e0, t0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle), time.perf_counter()
        amplified()
        torch.cuda.synchronize()
        energies.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) - e0)
        times.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.05)

    n = len(energies)
    mean = sum(energies) / n
    std = (sum((e - mean) ** 2 for e in energies) / max(n - 1, 1)) ** 0.5
    rel = std / mean * 100 if mean else float("inf")
    t_mean = sum(times) / n
    results.update(
        stability_mean_mj=round(mean, 2),
        stability_std_mj=round(std, 2),
        stability_rel_std_pct=round(rel, 2),
        stability_window_ms=round(t_mean, 2),
    )

    zeros = sum(1 for e in energies if e <= 0)
    results["stability_zero_readings"] = zeros
    say(OK, f"{reps} matmuls per trial took {t_mean:.1f} ms, "
            f"costing {mean:.0f} +/- {std:.0f} mJ"
            f"  ({mean / t_mean:.0f} W)")

    # Aliasing has a signature: zero readings, or a spread wider than the
    # mean. Report that as a measurement fault rather than blaming the GPU --
    # the two call for completely different responses.
    aliased = zeros > 0 or (period and t_mean < 3 * period)
    if aliased:
        say(FAIL, f"Window too short for this sensor -- {zeros}/{n} trials "
                  f"read 0 mJ",
            f"A {t_mean:.0f} ms window against a ~{period or 0:.0f} ms update "
            "period aliases: the spread measures the sampler, not the GPU. "
            "Re-run with a larger --reps; the auto-sizer should have handled "
            "this, so report it if you see it.")
    elif rel < 5:
        say(OK, f"Relative std dev {rel:.1f}% -- comfortably inside CLEAR's 10% bar")
    elif rel < 10:
        say(WARN, f"Relative std dev {rel:.1f}% -- inside the bar but with no margin",
            "Raise the repetition and trial counts for real measurements.")
    else:
        say(FAIL, f"Relative std dev {rel:.1f}% -- fails CLEAR's 10% bar",
            "The window is adequate, so this is the GPU: unlocked clocks, a "
            "co-tenant workload, or thermal throttling. Component-level energy "
            "measured here would not be publishable.")


# --------------------------------------------------------------- verdict --


def verdict() -> None:
    section("Verdict: what can run here")
    energy_ok = results.get("energy_counter") and results.get("energy_counter_ticks_mj", 0) > 0
    rel = results.get("stability_rel_std_pct")
    clean = results.get("other_gpu_processes", 0) == 0
    runnable = results.get("pairs_fp16") or results.get("pairs_nf4")
    vram_ok = bool(runnable)
    which = ", ".join(results.get("pairs_fp16") or results.get("pairs_nf4") or [])

    # Three states, not two. "Not measured" is not the same as "failed", and
    # reporting a skipped stability sweep as FAIL would condemn a machine
    # that was never tested -- exactly the wrong signal to act on.
    if not (energy_ok and clean):
        measurable: bool | None = False
        why = ("no usable energy counter" if not energy_ok
               else "another process is on this GPU")
    elif rel is None:
        measurable = None
        why = "stability not measured -- re-run without --quick"
    elif rel < 10:
        measurable = True
        why = f"relative std dev {rel:.1f}% is inside CLEAR's 10% bar"
    else:
        measurable = False
        why = f"relative std dev {rel:.1f}% fails CLEAR's 10% bar"

    rows: list[tuple[str, bool | None, str]] = [
        ("1  baseline draft-verify pipeline, correctness", vram_ok,
         f"runnable pairs: {which or 'none'}"),
        ("2  strategy sweep: acceptance, latency, throughput", vram_ok,
         "timing only; tolerates an unsteady clock"),
        ("3  CLEAR activation store (hooks -> cached tensors)", vram_ok,
         "no energy sensor involved; disk-bound"),
        ("4  CLEAR amplification: component-level energy", measurable, why),
        ("5  energy-optimal K* analysis", measurable,
         "only as trustworthy as phase 4"),
    ]
    for label, ok, note in rows:
        tag = WARN if ok is None else (OK if ok else FAIL)
        print(f"  [{tag}] {label}\n         {DIM}{note}{RESET}")

    results["phases_supported"] = {label.strip(): ok for label, ok, _ in rows}

    print()
    if all(ok for _, ok, _ in rows):
        print(f"  {GREEN}Everything runs here.{RESET} Lock the clocks before measuring and keep the GPU to yourself.")
    elif vram_ok and measurable is None:
        print(f"  {YELLOW}Phases 1-3 run here.{RESET} Whether phase 4 can is "
              f"untested:\n  re-run without --quick to measure the "
              f"stability bar (about 2 minutes).")
    elif vram_ok:
        print(
            f"  {YELLOW}Development machine, not a measurement machine.{RESET}\n"
            "  Build and debug phases 1-3 here; run phases 4-5 on a dedicated,\n"
            "  clock-lockable GPU."
        )
    else:
        print(f"  {RED}No Dutta et al. pair fits this GPU, even at NF4.{RESET}")


class _Tee:
    """Write to the console and to a file at once.

    The file copy has ANSI colour stripped: the console is a pty under
    Jupyter so the escapes are real, and they make a saved record awkward
    to read or diff later.
    """

    _ANSI = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, console, fh):
        self.console, self.fh = console, fh

    def write(self, data: str) -> int:
        self.console.write(data)
        self.fh.write(self._ANSI.sub("", data))
        return len(data)

    def flush(self) -> None:
        self.console.flush()
        self.fh.flush()

    def isatty(self) -> bool:
        return self.console.isatty()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the stability sweep")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--reps", type=int, default=None,
                    help="reps per trial; default auto-sizes to the "
                         "measured sensor update period")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    ap.add_argument("--text", metavar="PATH",
                    help="also write this whole report to a plain-text file, "
                         "so the run's provenance is saved alongside results")
    args = ap.parse_args()

    text_fh = None
    if args.text:
        pathlib.Path(args.text).parent.mkdir(parents=True, exist_ok=True)
        text_fh = open(args.text, "w")
        sys.stdout = _Tee(sys.__stdout__, text_fh)

    print("=" * 72)
    print("  Speculative decoding + CLEAR: environment probe")
    print("=" * 72)

    probe_host()
    handle = None
    if probe_gpu():
        probe_model_fit()
        probe_quantization()
        handle = probe_nvml()
        probe_clock_control(handle)
        probe_profilers()
        if not args.quick:
            probe_stability(handle, trials=args.trials, reps=args.reps)
    verdict()

    if args.json:
        # create the directory: the probe often runs before anything else has
        # made results/, and crashing at the last line after a full successful
        # probe is a pointlessly annoying way to lose the record
        import pathlib as _pl
        _pl.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n  wrote {args.json}")

    if text_fh:
        path = args.text
        sys.stdout = sys.__stdout__
        text_fh.close()
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
