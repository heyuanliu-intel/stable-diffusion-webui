import threading
import time
from collections import defaultdict

import torch
import habana_frameworks.torch as htorch
from habana_frameworks.torch.hpu import memory_stats


class MemUsageMonitor(threading.Thread):
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        self.name = name
        self.device = device
        self.opts = opts

        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        try:
            self.cuda_mem_get_info()
            torch.cuda.memory_stats(self.device)
        except Exception as e:  # AMD or whatever
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def cuda_mem_get_info(self):
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self.cuda_mem_get_info()[0]

            while self.run_flag.is_set():
                free, total = self.cuda_mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))

        print(self, 'raw torch memory stats:')
        tm = torch.cuda.memory_stats(self.device)
        for k, v in tm.items():
            if 'bytes' not in k:
                continue
            print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))

        print(torch.cuda.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = self.cuda_mem_get_info()
            self.data["free"] = free
            self.data["total"] = total

            torch_stats = torch.cuda.memory_stats(self.device)
            self.data["active"] = torch_stats["active.all.current"]
            self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
            self.data["reserved"] = torch_stats["reserved_bytes.all.current"]
            self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()


class HPUMemUsageMonitor(threading.Thread):
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        self.name = name
        self.device = device
        self.opts = opts

        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        try:
            memory_stats(self.device)
        except Exception as e:  # AMD or whatever
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            htorch.hpu.reset_peak_memory_stats()
            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self.read()["free"]

            while self.run_flag.is_set():
                self.data["min_free"] = min(self.data["min_free"], self.read()["free"])

                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))

        print(self, 'raw torch memory stats:')
        tm = self.read()
        for k, v in tm.items():
            if 'bytes' not in k:
                continue
            print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))

        print(htorch.hpu.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            state = memory_stats(self.device)
            total = state["Limit"]
            max_in_use = state["MaxInUse"]
            in_use = state["InUse"]
            free = total - in_use

            self.data["total"] = total
            self.data["active"] = in_use
            self.data["active_peak"] = max_in_use
            self.data["free"] = free
            self.data["reserved"] = 0
            self.data["reserved_peak"] = 0
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
