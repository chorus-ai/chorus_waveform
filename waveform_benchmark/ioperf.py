import math
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import tracemalloc
import time
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
                      
      'max_rss': Maximum resident set size of the process during
                        execution, in MB
      'mem_used': Memory allocated by the process during execution, in MB
      
                      
    If the 'clear_cache' argument is true, then attempt to clear the
    system cache whenever a file is opened (so that 'n_bytes_read'
    should reflect the total number of bytes retrieved.)  Note that
    'clear_cache' *only* affects files that are opened through Python
    (using the open() or os.open() functions) and depends on the
    filesystem (it won't work on tmpfs, for instance.)
    
    several methods have been tested:
        rusage, tracemalloc, psutil, memory_profiler
        each is reporting different numbers.
        
        memory_profiler by default uses psutil.
        rusage can only report maxrss
        tracemalloc is reporting allocations.
        
        we will report a composite
    """
    def __init__(self, clear_cache=True, mem_profile = False):
        self._clear_cache = clear_cache
        self.mem_profile = mem_profile
        self.n_read_calls = 0
        self.n_seek_calls = 0
        self.n_bytes_read = 0
        self.cpu_seconds = 0
        if (self.mem_profile):
            tracemalloc.start()


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

        if (self.mem_profile):
            tracemalloc.reset_peak()

        self._walltime_start = time.time()

        # Measure past resource usage for this process and any children.
        self._rusage_self = resource.getrusage(resource.RUSAGE_SELF)
        self._rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Measure current resource usage for this process and any children.
        rusage_self = resource.getrusage(resource.RUSAGE_SELF)
        rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)

        # Calculate wall time
        self.walltime = time.time() - self._walltime_start    
        
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

        if (self.mem_profile):
            (final_usage, peak_usage) = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.malloced = (peak_usage - final_usage) / (2**20)
            self.max_rss = (rusage_self.ru_maxrss + rusage_children.ru_maxrss) / float(2**10)
        

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
