"""Linux child entrypoint: stop the service child if its supervisor disappears."""
import ctypes
import os
import signal
import sys


def main() -> None:
    parent_pid = int(sys.argv[1])
    command = sys.argv[2:]
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:
        raise SystemExit("Supervisor exited before child initialization")
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
