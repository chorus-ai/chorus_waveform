import os
import re
import resource
import signal
import subprocess
import sys
import threading


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

    If the 'clear_cache' argument is true, then attempt to clear the
    system cache whenever a file is opened (so that 'n_bytes_read'
    should reflect the total number of bytes retrieved.)  Note that
    'clear_cache' *only* affects files that are opened through Python
    (using the open() or os.open() functions) and depends on the
    filesystem (it won't work on tmpfs, for instance.)
    """
    def __init__(self, clear_cache=True):
        self._clear_cache = clear_cache
        self.n_read_calls = 0
        self.n_seek_calls = 0
        self.n_bytes_read = 0
        self.cpu_seconds = 0

    def __enter__(self):
        if self._clear_cache:
            with _nocache_lock:
                _nocache_enabled.add(self)

        # Run strace to track system calls.
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

        # Measure past resource usage for this process and any children.
        self._rusage_self = resource.getrusage(resource.RUSAGE_SELF)
        self._rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Measure current resource usage for this process and any children.
        rusage_self = resource.getrusage(resource.RUSAGE_SELF)
        rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)

        with _nocache_lock:
            _nocache_enabled.discard(self)

        # Stop the strace process and read its output.
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


_nocache_lock = threading.Lock()
_nocache_enabled = set()
_nocache_loop = False
sys.addaudithook(_nocache_hook)
