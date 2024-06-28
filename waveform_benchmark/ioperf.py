import math
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import tracemalloc
import psutil

class PerformanceCounter:
    """
    Context manager to measure CPU and I/O usage.

    This object has the following attributes, which (approximately)
    represent the system resources that are used during execution of
    the context manager - that is to say, anything that happens before
    or after the "with" block doesn't count.

      'n_read_calls': Number of times the read() system call was
                      invoked.

      'n_seek_calls': Number of times the lseek() system call was
                      invoked.

      'n_bytes_read': Number of bytes that were read from disk.

      'cpu_seconds':  Number of seconds that the CPU was busy
                      (multiplied by the number of CPU cores in use.)
                      
      'max_memory':       Memory usage in megabytes.

    If the 'clear_cache' argument is true, then attempt to clear the
    system cache whenever a file is opened (so that 'n_bytes_read'
    should reflect the total number of bytes retrieved.)  Note that
    'clear_cache' *only* affects files that are opened through Python
    (using the open() or os.open() functions) and depends on the
    filesystem (it won't work on tmpfs, for instance.)
    
    several methods have been tested:
        1 is rusage
        2 is tracemalloc
        3 is psutils
        0 is measurement outside of this function (e.g. memory_profiler).
        each is reporting different numbers...  for now rusage is used. 
    """
    def __init__(self, clear_cache=True, mem_method = 0):
        self._clear_cache = clear_cache
        self.n_read_calls = 0
        self.n_seek_calls = 0
        self.n_bytes_read = 0
        self.cpu_seconds = 0
        self.curr_memory = 0
        self.max_memory = 0
        self.mem_method = mem_method

    def __enter__(self):
        if self._clear_cache:
            with _nocache_lock:
                _nocache_enabled.add(self)

        if self._clear_cache and not _nocache_supported:
            self.n_bytes_read = math.nan

        # Run strace to track system calls.
        try:
            self._strace = subprocess.Popen(['strace', '-c', '-f',
                                             '-U', 'name,calls',
                                             '-e', 'lseek,read',
                                             '-p', str(os.getpid())],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT)
            # Wait for strace to attach to the current process.  (Subtract
            # one from n_read_calls because this counts as one.)
            self._strace.stdout.readline()
            self.n_read_calls -= 1
        except FileNotFoundError:
            self._strace = None
            self.n_read_calls = math.nan
            self.n_seek_calls = math.nan

        # Measure past resource usage for this process and any children.
        self._rusage_self = resource.getrusage(resource.RUSAGE_SELF)
        self._rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)

        if self.mem_method == 2:
            tracemalloc.start()
            (self._curr_memory, self._max_memory) = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
        elif self.mem_method == 3:
            self._process = psutil.Process()
            self._init_mem = self._process.memory_info().rss / float(2**20)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Measure current resource usage for this process and any children.
        rusage_self = resource.getrusage(resource.RUSAGE_SELF)
        rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)

        with _nocache_lock:
            _nocache_enabled.discard(self)

        # Stop the strace process and read its output.
        if self._strace is not None:
            self._strace.send_signal(signal.SIGINT)
            strace_data, _ = self._strace.communicate()
            for m in re.finditer(rb'^\s*(\w+)\s+(\d+)\s*$',
                                 strace_data, re.MULTILINE):
                if m.group(1) == b'read':
                    self.n_read_calls += int(m.group(2))
                elif m.group(1) == b'lseek':
                    self.n_seek_calls += int(m.group(2))

        # Calculate number of bytes read.  ru_inblock is always
        # measured in 512-byte blocks regardless of I/O block size.
        self.n_bytes_read += 512 * (rusage_self.ru_inblock
                                    + rusage_children.ru_inblock
                                    - self._rusage_self.ru_inblock
                                    - self._rusage_children.ru_inblock)

        # Calculate CPU seconds, including both user and kernel time.
        self.cpu_seconds += (rusage_self.ru_utime
                             + rusage_self.ru_stime
                             + rusage_children.ru_utime
                             + rusage_children.ru_stime
                             - self._rusage_self.ru_utime
                             - self._rusage_self.ru_stime
                             - self._rusage_children.ru_utime
                             - self._rusage_children.ru_stime)
        
        if self.mem_method == 1:
            self.max_memory = (rusage_self.ru_maxrss) / float(2**10)
        elif self.mem_method == 2:
            (self.curr_memory, self.max_memory) = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        elif self.mem_method == 3:
            self.curr_memory = self._process.memory_info().rss / float(2**20)
            self.max_memory = self._process.memory_info().vms / float(2**20) - self._init_mem

_nocache_lock = threading.Lock()
_nocache_supported = False
_nocache_enabled = set()
_nocache_loop = False

if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
    _nocache_supported = True

    def _nocache_hook(event, args):
        global _nocache_loop
        if event == 'open' and not _nocache_loop:
            with _nocache_lock:
                if _nocache_enabled and isinstance(args[0], (str, bytes)):
                    _nocache_loop = True
                    fd = None
                    try:
                        fd = os.open(args[0], os.O_RDONLY | os.O_CLOEXEC)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    except OSError:
                        pass
                    finally:
                        if fd is not None:
                            os.close(fd)
                        _nocache_loop = False

    sys.addaudithook(_nocache_hook)
