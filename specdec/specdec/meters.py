"""Phase-level timing and (optional) GPU energy instrumentation.

This is deliberately coarse. ``nvmlDeviceGetTotalEnergyConsumption`` has
millijoule resolution but the NVML power sensor updates on a millisecond
cadence, so a single draft or verify phase lasting tens of microseconds is
*below the sensor's noise floor*. The numbers this module produces are only
trustworthy when aggregated over a whole generation run.

Component-level energy -- which component inside draft or verify burns the
joules -- is exactly what CLEAR's amplification stage exists to answer, and
that lands in a later phase of the project. The hook points here
(``PhaseMeter.phase``) are where that instrumentation will attach.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

import torch

try:  # pragma: no cover - depends on host
    import pynvml

    _PYNVML_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    pynvml = None
    _PYNVML_IMPORT_ERROR = exc


class EnergyProbe:
    """Thin wrapper over NVML's cumulative energy counter.

    ``supported`` is False when pynvml is missing, no GPU is present, or the
    device predates Volta (the energy counter needs compute capability >= 7.0;
    a Kaggle P100 at 6.0 will raise ``NVML_ERROR_NOT_SUPPORTED`` here).
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.nvml_index: Optional[int] = None
        self.supported = False
        self.reason = ""
        self._handle = None

        if pynvml is None:
            self.reason = f"pynvml unavailable: {_PYNVML_IMPORT_ERROR}"
            return
        try:
            pynvml.nvmlInit()
            self._handle, self.nvml_index = self._resolve_handle(device_index)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except Exception as exc:
            self.reason = f"NVML energy counter unavailable: {exc}"
            self._handle = None
            return
        self.supported = True

    @staticmethod
    def _resolve_handle(device_index: int):
        """Find the NVML handle for the GPU torch is actually using.

        NVML enumerates every physical GPU and ignores CUDA_VISIBLE_DEVICES,
        so NVML index 0 is not torch's device 0 whenever that variable is set
        -- and CUDA_DEVICE_ORDER can reorder things further. Two processes
        pinned to different GPUs with CUDA_VISIBLE_DEVICES=0 and =1 would
        both read physical GPU 0 and silently report the same card's energy,
        one of them for a workload it never ran.

        Matching on UUID is the only binding that survives both remappings.
        """
        uuid_hex = None
        try:
            raw = getattr(torch.cuda.get_device_properties(device_index),
                          "uuid", None)
            if raw is not None:
                uuid_hex = str(raw).replace("-", "").lower().replace("gpu", "")
        except Exception:
            pass

        count = pynvml.nvmlDeviceGetCount()
        if uuid_hex:
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                nv = pynvml.nvmlDeviceGetUUID(handle)
                nv = nv.decode() if isinstance(nv, bytes) else nv
                if nv.replace("-", "").lower().replace("gpu", "") == uuid_hex:
                    return handle, i

        # UUID unavailable: fall back to CUDA_VISIBLE_DEVICES, which is right
        # whenever NVML's order matches CUDA's (the usual case).
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            parts = [p.strip() for p in visible.split(",") if p.strip()]
            if device_index < len(parts) and parts[device_index].isdigit():
                idx = int(parts[device_index])
                if idx < count:
                    return pynvml.nvmlDeviceGetHandleByIndex(idx), idx

        return pynvml.nvmlDeviceGetHandleByIndex(device_index), device_index

    def read_mj(self) -> Optional[float]:
        """Cumulative GPU energy in millijoules since the driver last loaded."""
        if not self.supported:
            return None
        try:
            return float(
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            )
        except Exception:
            return None

    def measure_idle_w(self, seconds: float = 3.0) -> Optional[float]:
        """Static draw with the GPU doing nothing.

        Every phase reading includes this. A draft phase that takes 8 ms is
        charged ~8 ms of the card merely being powered on, so raw phase
        energy conflates work with standing cost -- and because draft phases
        get longer as K grows, some of the apparent "draft energy rises with
        K" is just static power multiplied by time. Recording the idle draw
        once lets the analysis report energy *above* idle alongside the raw
        figure, which is the quantity that actually attributes to the work.
        """
        if not self.supported:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(0.5)  # let clocks settle after whatever ran before
        e0 = self.read_mj()
        t0 = time.perf_counter()
        time.sleep(seconds)
        e1 = self.read_mj()
        dt = time.perf_counter() - t0
        if e0 is None or e1 is None or dt <= 0:
            return None
        return (e1 - e0) / 1000.0 / dt  # mJ -> J, per second = W

    def shutdown(self) -> None:
        if self.supported:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self.supported = False


class PhaseMeter:
    """Accumulates wall time (and energy, where available) per named phase."""

    def __init__(self, energy: Optional[EnergyProbe] = None, sync: bool = True):
        self.energy = energy
        self.sync = sync and torch.cuda.is_available()
        self.time_ms: dict[str, float] = defaultdict(float)
        self.energy_mj: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def _sync(self) -> None:
        if self.sync:
            torch.cuda.synchronize()

    @contextmanager
    def phase(self, name: str):
        self._sync()
        e0 = self.energy.read_mj() if self.energy else None
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = (time.perf_counter() - t0) * 1000.0
            self.time_ms[name] += dt
            self.calls[name] += 1
            if e0 is not None:
                e1 = self.energy.read_mj()
                if e1 is not None:
                    self.energy_mj[name] += e1 - e0

    def summary(self) -> dict:
        out: dict = {"time_ms": dict(self.time_ms), "calls": dict(self.calls)}
        if self.energy and self.energy.supported:
            out["energy_mj"] = dict(self.energy_mj)
        return out

    def reset(self) -> None:
        self.time_ms.clear()
        self.energy_mj.clear()
        self.calls.clear()
