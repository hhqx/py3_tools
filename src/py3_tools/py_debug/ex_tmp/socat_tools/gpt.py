import os
import pty
import subprocess
import tempfile
import logging
import shutil
import signal

class UnixTerminalSocketServer:
    def __init__(self, socket_path="/tmp/terminal.sock", shell="/bin/bash", log_level=logging.INFO):
        self.socket_path = socket_path
        self.shell = shell
        self.logger = logging.getLogger("UnixTerminalSocketServer")
        self.logger.setLevel(log_level)

        self.workdir = tempfile.mkdtemp(prefix="term_socket_")
        self.in_pipe = os.path.join(self.workdir, "in.pipe")
        self.out_pipe = os.path.join(self.workdir, "out.pipe")

        self.child_pid = None
        self.socat_proc = None

    def _create_pipes(self):
        os.mkfifo(self.in_pipe)
        os.mkfifo(self.out_pipe)
        self.logger.info(f"Created pipes: {self.in_pipe}, {self.out_pipe}")

    def _spawn_shell_with_pty(self):
        pid, fd = pty.fork()
        if pid == 0:
            # child
            with open(self.in_pipe, "rb") as infd, open(self.out_pipe, "wb") as outfd:
                os.dup2(infd.fileno(), 0)  # stdin
                os.dup2(outfd.fileno(), 1)  # stdout
                os.dup2(outfd.fileno(), 2)  # stderr
                os.execvp(self.shell, [self.shell])
        else:
            self.child_pid = pid
            self.logger.info(f"Spawned shell PID: {pid}")

    def _launch_socat(self):
        # Ensure no leftover socket
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        socat_cmd = f"socat UNIX-LISTEN:{self.socket_path},fork SYSTEM:'cat {self.out_pipe} & cat > {self.in_pipe}'"
        self.logger.info(f"Launching socat: {socat_cmd}")
        self.socat_proc = subprocess.Popen(socat_cmd, shell=True, preexec_fn=os.setsid)

    def start(self):
        self._create_pipes()
        self._spawn_shell_with_pty()
        self._launch_socat()
        self.logger.info("Server started. Connect via: socat - UNIX-CONNECT:/tmp/terminal.sock")

    def stop(self):
        if self.socat_proc:
            os.killpg(os.getpgid(self.socat_proc.pid), signal.SIGTERM)
            self.logger.info("Socat process terminated.")
        if self.child_pid:
            os.kill(self.child_pid, signal.SIGTERM)
            self.logger.info("Shell process terminated.")
        shutil.rmtree(self.workdir)
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        self.logger.info("Cleaned up resources.")

# ========== TEST ==========

def test_unix_terminal_socket_server():
    import time

    server = UnixTerminalSocketServer(log_level=logging.DEBUG)
    try:
        server.start()
        print("Server running. Try connecting via:")
        print(f"  socat - UNIX-CONNECT:{server.socket_path}")
        print("Sleeping for 60s to allow manual testing...")
        time.sleep(60)
    finally:
        server.stop()

if __name__ == "__main__":
    test_unix_terminal_socket_server()
