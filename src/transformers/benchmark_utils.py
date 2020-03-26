"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import linecache
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

from .file_utils import is_tf_available, is_torch_available


if is_torch_available():
    from torch.cuda import empty_cache as torch_empty_cache
if is_tf_available():
    from tensorflow.python.eager import context as tf_context


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


_is_memory_tracing_enabled = False


def is_memory_tracing_enabled():
    global _is_memory_tracing_enabled
    return _is_memory_tracing_enabled


@dataclass(frozen=True)
class Frame:
    """ `Frame` is used to gather the current frame state:
        - 'filename' (string): Name of the file currently executed
        - 'module' (string): Name of the module currently executed
        - 'line_number' (int): Number of the line currently executed
        - 'event' (string): Event that triggered the tracing (default will be "line")
        - 'line_text' (string): Text of the line in the python script
    """

    filename: str
    module: str
    line_number: int
    event: str
    line_text: str


@dataclass
class MemoryState:
    """ `MemoryState` lists frame + CPU/GPU memory:
        - `cpu`: CPU memory at or before the current frame as a `Memory` named tuple
        - `gpu`: GPU memory at or before during the current frame as a `Memory` named tuple
        - `frame` (`Frame`): the current frame
        Also provide a few properties:
            `cpu_gpu`: sum of the CPU + GPU memory at or before during the current frame as a `Memory` named tuple
            `cpu_with_units`: CPU memory as a human readable string
            `gpu_with_units`: GPU memory as a human readable string
            `cpu_gpu_with_units`: CPU+GPU memory as a human readable string
    """

    cpu: int
    gpu: int
    frame: Optional[Frame] = None

    @property
    def cpu_gpu(self) -> int:
        return self.cpu + self.gpu

    @property
    def cpu_with_units(self) -> str:
        return bytes_to_human_readable(self.cpu)

    @property
    def gpu_with_units(self) -> str:
        return bytes_to_human_readable(self.gpu)

    @property
    def cpu_gpu_with_units(self) -> str:
        return bytes_to_human_readable(self.cpu + self.gpu)


@dataclass
class MemorySummary:
    """ `MemorySummary` namedtuple otherwise with the fields:
        - `absolute_mem_list`: total CPU/GPU memory used at each line
            a list of `MemoryState` namedtuple (see below)
        - `relative_mem_list`: relative difference in CPU/GPU memory at each line
            a list of `MemoryState` namedtuple (see below) computed from the provided `memory_trace`
            by substracting the memory after executing each line from the memory before executing said line.
        - `absolute_mem_sorted`: total CPU/GPU memory used at each line sorted by lines (max among all the times a line is executed)
            a list of `MemoryState` namedtuple (see below)
            The list is sorted from the frame with the largest memory consumption to the frame with the smallest (can be negative if memory is released)
        - `relative_mem_sorted`: relative difference in CPU/GPU memory sorted by lines (cumulative increase among all the times a line is executed)
            a list of `MemoryState` namedtuple (see below) with cumulative increase in memory for each line
            obtained by summing repeted memory increase for a line if it's executed several times.
            The list is sorted from the frame with the largest memory consumption to the frame with the smallest (can be negative if memory is released)
        - `total`: total memory increase during the full tracing as a `Memory` named tuple (see below).
            Line with memory release (negative consumption) are ignored if `ignore_released_memory` is `True` (default).
    """

    absolute_mem_list: List[MemoryState]
    relative_mem_list: List[MemoryState]
    absolute_mem_sorted: List[MemoryState]
    relative_mem_sorted: List[MemoryState]
    relative_mem_total: MemoryState


MemoryTrace = List[MemoryState]


def start_memory_tracing(
    modules_to_trace: Optional[Union[str, Iterable[str]]] = None,
    modules_not_to_trace: Optional[Union[str, Iterable[str]]] = None,
    events_to_trace: str = "line",
    gpus_to_trace: Optional[List[int]] = None,
) -> MemoryTrace:
    """ Setup line-by-line tracing to record rss mem (RAM) at each line of a module or sub-module.
        See `../../examples/benchmarks.py for a usage example.
        Current memory consumption is returned using psutil and in particular is the RSS memory
            "Resident Set Size” (the non-swapped physical memory the process is using).
            See https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

        Args:
            - `modules_to_trace`: (None, string, list/tuple of string)
                if None, all events are recorded
                if string or list of strings: only events from the listed module/sub-module will be recorded (e.g. 'fairseq' or 'transformers.modeling_gpt2')
            - `modules_not_to_trace`: (None, string, list/tuple of string)
                if None, no module is avoided
                if string or list of strings: events from the listed module/sub-module will not be recorded (e.g. 'torch')
            - `events_to_trace`: string or list of string of events to be recorded (see official python doc for `sys.settrace` for the list of events)
                default to line
            - `gpus_to_trace`: (optional list, default None) list of GPUs to trace. Default to tracing all GPUs

        Return:
            - `memory_trace` is a list of `MemoryState` for each event (default each line of the traced script).
                - `MemoryState` are simple classes with the following attributes:
                    - 'frame': a `Frame` namedtuple (see below) storing information on the current tracing frame (current file, location in current file)
                    - 'cpu': CPU RSS memory state *before* executing the line
                    - 'gpu': GPU used memory *before* executing the line (sum for all GPUs or for only `gpus_to_trace` if provided)
                    - `cpu_gpu`: CPU + GPU memory *before* executing the line

        `Frame` is a namedtuple used by `MemoryState` to list the current frame state.
            `Frame` has the following fields:
            - 'filename' (string): Name of the file currently executed
            - 'module' (string): Name of the module currently executed
            - 'line_number' (int): Number of the line currently executed
            - 'event' (string): Event that triggered the tracing (default will be "line")
            - 'line_text' (string): Text of the line in the python script

    """
    try:
        import psutil
    except (ImportError):
        logger.warning(
            "Psutil not installed, we won't log CPU memory usage. "
            "Install psutil (pip install psutil) to use CPU memory tracing."
        )
        process = None
    else:
        process = psutil.Process(os.getpid())

    try:
        from py3nvml import py3nvml

        py3nvml.nvmlInit()
        devices = list(range(py3nvml.nvmlDeviceGetCount())) if gpus_to_trace is None else gpus_to_trace
        py3nvml.nvmlShutdown()
    except ImportError:
        logger.warning(
            "py3nvml not installed, we won't log GPU memory usage. "
            "Install py3nvml (pip install py3nvml) to use GPU memory tracing."
        )
        log_gpu = False
    except (OSError, py3nvml.NVMLError):
        logger.warning("Error while initializing comunication with GPU. " "We won't perform GPU memory tracing.")
        log_gpu = False
    else:
        log_gpu = is_torch_available() or is_tf_available()

    memory_trace = []

    def traceit(frame, event, args):
        """ Tracing method executed before running each line in a module or sub-module
            Record memory allocated in a list with debugging information
        """
        global _is_memory_tracing_enabled

        if not _is_memory_tracing_enabled:
            return traceit

        # Filter events
        if events_to_trace is not None:
            if isinstance(events_to_trace, str) and event != events_to_trace:
                return traceit
            elif isinstance(events_to_trace, (list, tuple)) and event not in events_to_trace:
                return traceit

        # Filter modules
        name = frame.f_globals["__name__"]
        if not isinstance(name, str):
            return traceit
        else:
            # Filter whitelist of modules to trace
            if modules_to_trace is not None:
                if isinstance(modules_to_trace, str) and modules_to_trace not in name:
                    return traceit
                elif isinstance(modules_to_trace, (list, tuple)) and all(m not in name for m in modules_to_trace):
                    return traceit

            # Filter blacklist of modules not to trace
            if modules_not_to_trace is not None:
                if isinstance(modules_not_to_trace, str) and modules_not_to_trace in name:
                    return traceit
                elif isinstance(modules_not_to_trace, (list, tuple)) and any(m in name for m in modules_not_to_trace):
                    return traceit

        # Record current tracing state (file, location in file...)
        lineno = frame.f_lineno
        filename = frame.f_globals["__file__"]
        if filename.endswith(".pyc") or filename.endswith(".pyo"):
            filename = filename[:-1]
        line = linecache.getline(filename, lineno).rstrip()
        traced_state = Frame(filename, name, lineno, event, line)

        # Record current memory state (rss memory) and compute difference with previous memory state
        cpu_mem = 0
        if process is not None:
            mem = process.memory_info()
            cpu_mem = mem.rss

        gpu_mem = 0
        if log_gpu:
            # Clear GPU caches
            if is_torch_available():
                torch_empty_cache()
            if is_tf_available():
                tf_context.context()._clear_caches()  # See https://github.com/tensorflow/tensorflow/issues/20218#issuecomment-416771802

            # Sum used memory for all GPUs
            py3nvml.nvmlInit()
            for i in devices:
                handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem += meminfo.used
            py3nvml.nvmlShutdown()

        mem_state = MemoryState(cpu_mem, gpu_mem, traced_state)
        memory_trace.append(mem_state)

        return traceit

    sys.settrace(traceit)

    global _is_memory_tracing_enabled
    _is_memory_tracing_enabled = True

    return memory_trace


def stop_memory_tracing(
    memory_trace: Optional[MemoryTrace] = None, ignore_released_memory: bool = True
) -> Optional[MemorySummary]:
    """ Stop memory tracing cleanly and return a summary of the memory trace if a trace is given.

        Args:
            - `memory_trace` (optional output of start_memory_tracing, default: None): memory trace to convert in summary
            - `ignore_released_memory` (boolean, default: None): if True we only sum memory increase to compute total memory

        Return:
            - None if `memory_trace` is None
            - `MemorySummary` namedtuple otherwise with the fields:
                - `sequential`: a list of `MemoryState` namedtuple (see below) computed from the provided `memory_trace`
                    by substracting the memory after executing each line from the memory before executing said line.
                - `cumulative`: a list of `MemoryState` namedtuple (see below) with cumulative increase in memory for each line
                    obtained by summing repeted memory increase for a line if it's executed several times.
                    The list is sorted from the frame with the largest memory consumption to the frame with the smallest (can be negative if memory is released)
                - `total`: total memory increase during the full tracing as a `Memory` named tuple (see below).
                    Line with memory release (negative consumption) are ignored if `ignore_released_memory` is `True` (default).

        `Memory` named tuple have fields
            - `byte` (integer): number of bytes,
            - `string` (string): same as human readable string (ex: "3.5MB")

        `Frame` are namedtuple used to list the current frame state and have the following fields:
            - 'filename' (string): Name of the file currently executed
            - 'module' (string): Name of the module currently executed
            - 'line_number' (int): Number of the line currently executed
            - 'event' (string): Event that triggered the tracing (default will be "line")
            - 'line_text' (string): Text of the line in the python script

        `MemoryState` are namedtuples listing frame + CPU/GPU memory with the following fields:
            - `frame` (`Frame`): the current frame (see above)
            - `cpu`: CPU memory consumed at during the current frame as a `Memory` named tuple
            - `gpu`: GPU memory consumed at during the current frame as a `Memory` named tuple
            - `cpu_gpu`: CPU + GPU memory consumed at during the current frame as a `Memory` named tuple
    """
    global _is_memory_tracing_enabled
    _is_memory_tracing_enabled = False

    if memory_trace is not None and len(memory_trace) > 1:
        init_mem = memory_trace[0]
        absolute_mem_list = []
        relative_mem_list = []
        absolute_mem_dict = defaultdict(lambda: [])
        relative_mem_dict = defaultdict(lambda: [])
        for line, next_line in zip(memory_trace[:-1], memory_trace[1:]):
            absolute_mem = MemoryState(line.cpu - init_mem.cpu, line.gpu - init_mem.gpu, line.frame)
            relative_mem = MemoryState(next_line.cpu - line.cpu, next_line.gpu - line.gpu, line.frame)
            absolute_mem_list.append(absolute_mem)
            relative_mem_list.append(relative_mem)
            absolute_mem_dict[line.frame].append(absolute_mem)
            relative_mem_dict[line.frame].append(relative_mem)

        relative_mem_sorted = list(MemoryState(sum(v.cpu for v in l), sum(v.gpu for v in l), k) for k, l in relative_mem_dict.items())
        absolute_mem_sorted = list(MemoryState(max(v.cpu for v in l), max(v.gpu for v in l), k) for k, l in absolute_mem_dict.items())

        relative_mem_sorted = sorted(relative_mem_sorted, key=lambda x: x.cpu_gpu, reverse=True)
        absolute_mem_sorted = sorted(absolute_mem_sorted, key=lambda x: x.cpu_gpu, reverse=True)

        to_sum = (
            filter(lambda m: m.cpu_gpu > 0, relative_mem_list)
            if ignore_released_memory
            else relative_mem_list
        )
        relative_mem_total = MemoryState(sum(v.cpu for v in to_sum), sum(v.gpu for v in to_sum))
        return MemorySummary(
            absolute_mem_list=absolute_mem_list,
            relative_mem_list=relative_mem_list,
            relative_mem_sorted=relative_mem_sorted,
            absolute_mem_sorted=absolute_mem_sorted,
            relative_mem_total=relative_mem_total,
        )

    return None


def bytes_to_human_readable(memory_amount: int) -> str:
    """ Utility to convert a number of bytes (int) in a human readable string (with units)
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if memory_amount > -1024.0 and memory_amount < 1024.0:
            return "{:.3f}{}".format(memory_amount, unit)
        memory_amount /= 1024.0
    return "{:.3f}TB".format(memory_amount)
