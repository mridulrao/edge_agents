"""
Edge Question Generation Pipeline - LlamaServer Manager
Manages llama-server process lifecycle and health monitoring
"""

import asyncio
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import aiohttp
import logging

import psutil 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------
# Memory helpers
# ----------------------------

def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def _proc_mem_mb(pid: int) -> Dict[str, float | None]:
    """
    Best-effort process memory stats in MB.
    rss_mb is always available.
    uss_mb may be unavailable on some platforms.
    """
    p = psutil.Process(pid)
    mi = p.memory_info()

    out: Dict[str, float | None] = {
        "rss_mb": _bytes_to_mb(mi.rss),
        "vms_mb": _bytes_to_mb(getattr(mi, "vms", 0)),
        "uss_mb": None,
    }

    try:
        mfi = p.memory_full_info()
        if hasattr(mfi, "uss"):
            out["uss_mb"] = _bytes_to_mb(mfi.uss)
    except Exception:
        pass

    return out


@dataclass
class ServerConfig:
    """Configuration for llama-server instance"""

    # Model settings
    model_path: str

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080

    # Context settings
    ctx_size: int = 1048
    n_predict: int = 512

    # Hardware acceleration
    n_gpu_layers: int = 0  # 0 = CPU only, -1 = all layers on GPU

    # Threading
    n_threads: int = 4

    # Memory settings
    no_mmap: bool = False  # Disable memory mapping
    mlock: bool = False    # Lock model in RAM

    # Additional args
    extra_args: list = None

    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = []

    def to_cmd_args(self, llama_server_path: str = "llama-server", is_llama_cli: bool = False) -> list:
        """Convert config to command-line arguments"""
        args = [llama_server_path]

        # Add --server flag for llama-cli
        if is_llama_cli:
            args.append("--server")

        args.extend([
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.ctx_size),
            "--n-predict", str(self.n_predict),
            "--threads", str(self.n_threads),
        ])

        # GPU layers
        if self.n_gpu_layers != 0:
            args.extend(["--n-gpu-layers", str(self.n_gpu_layers)])

        # Memory settings
        if self.no_mmap:
            args.append("--no-mmap")

        if self.mlock:
            args.append("--mlock")

        # Add extra args
        args.extend(self.extra_args)

        return args


class LlamaServerManager:
    """
    Manages llama-server process lifecycle.

    Handles:
    - Starting/stopping server process
    - Health monitoring
    - Auto-restart on failure
    - Graceful shutdown
    """

    def __init__(
        self,
        config: ServerConfig,
        llama_server_path: str = "models/llama.cpp/build/bin/llama-server",
        startup_timeout: float = 120.0,
        health_check_interval: float = 1.0,
    ):
        """
        Initialize server manager.

        Args:
            config: Server configuration
            llama_server_path: Path to llama-server binary (or llama-cli)
            startup_timeout: Max seconds to wait for server startup
            health_check_interval: Seconds between health checks
        """
        self.config = config
        self.llama_server_path = llama_server_path
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval

        self.process: Optional[subprocess.Popen] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._shutdown_requested = False
        self._is_llama_cli = False  # Track if we're using llama-cli

        # NEW: last start/load stats for external consumers (profiling scripts)
        self.last_start_stats: Optional[Dict[str, Any]] = None

    @property
    def base_url(self) -> str:
        """Get server base URL"""
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def is_running(self) -> bool:
        """Check if process is running"""
        return self.process is not None and self.process.poll() is None

    @property
    def pid(self) -> Optional[int]:
        """PID of the llama-server process (if spawned)."""
        return self.process.pid if self.process is not None else None

    async def _ensure_session(self):
        """Create aiohttp session if needed"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def _close_session(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _validate_config(self):
        """Validate configuration before starting"""
        # Check model file exists
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

        if model_path.suffix != ".gguf":
            logger.warning(f"Model file does not have .gguf extension: {model_path}")

        # Try to find a working llama binary
        binaries_to_try = [
            self.llama_server_path,
            "llama-server",
            "llama-cli",
            "./llama-server",
            "./llama-cli",
            "build/bin/llama-server",
            "build/bin/llama-cli",
        ]

        working_binary = None

        for binary in binaries_to_try:
            try:
                result = subprocess.run(
                    [binary, "--version"],
                    capture_output=True,
                    timeout=10,
                    text=True,
                )
                if result.returncode == 0:
                    working_binary = binary
                    # Check if this is llama-cli
                    if "llama-cli" in binary.lower() or "cli" in result.stdout.lower():
                        self._is_llama_cli = True
                        logger.info(f"Using llama-cli as server: {binary}")
                    else:
                        logger.info(f"Using llama-server: {binary}")
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            except Exception:
                continue

        if working_binary is None:
            raise FileNotFoundError(
                "No working llama binary found. Tried:\n"
                + "\n".join(f"  - {b}" for b in binaries_to_try)
                + "\n\nTo install llama.cpp:\n"
                  "  git clone https://github.com/ggerganov/llama.cpp.git\n"
                  "  cd llama.cpp\n"
                  "  mkdir build && cd build\n"
                  "  cmake .. -DGGML_METAL=ON  # For macOS\n"
                  "  cmake --build . --config Release\n"
                  "  # Binary will be in build/bin/llama-server or build/bin/llama-cli"
            )

        # Update the path to the working binary
        self.llama_server_path = working_binary

    async def start(self) -> bool:
        """
        Start llama-server process.

        Returns:
            True if server started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Server is already running")
            return True

        # Validate configuration
        try:
            self._validate_config()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

        cmd = self.config.to_cmd_args(self.llama_server_path, self._is_llama_cli)
        logger.info(f"Starting {'llama-cli' if self._is_llama_cli else 'llama-server'}: {' '.join(cmd)}")

        # Start process
        try:
            t_spawn0 = time.perf_counter()
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            t_spawn1 = time.perf_counter()
        except Exception as e:
            logger.error(f"Failed to start server process: {e}")
            return False

        pid = self.pid
        logger.info(f"Spawned server PID={pid} (spawn_latency_ms={(t_spawn1 - t_spawn0) * 1000.0:.1f})")

        # Wait for ready + track load latency + memory peaks during load
        logger.info(f"Waiting for server to become ready (timeout: {self.startup_timeout}s)...")

        await self._ensure_session()

        t_load0 = time.perf_counter()
        start_time = time.time()

        rss_peak = 0.0
        vms_peak = 0.0
        uss_peak: Optional[float] = None

        # Loop until healthy or timeout
        while time.time() - start_time < self.startup_timeout:
            # If process died, log stderr and fail
            if not self.is_running:
                logger.error("Server process died during startup")
                if self.process and self.process.stderr:
                    try:
                        stderr = self.process.stderr.read()
                        if stderr:
                            logger.error(f"Server stderr: {stderr}")
                    except Exception:
                        pass
                await self.stop()
                self.last_start_stats = {
                    "ok": False,
                    "pid": pid,
                    "error": "process_died_during_startup",
                }
                return False

            # Sample memory peak
            if pid is not None:
                try:
                    m = _proc_mem_mb(pid)
                    rss_peak = max(rss_peak, float(m["rss_mb"] or 0.0))
                    vms_peak = max(vms_peak, float(m["vms_mb"] or 0.0))
                    if m["uss_mb"] is not None:
                        uss_peak = m["uss_mb"] if uss_peak is None else max(uss_peak, m["uss_mb"])
                except Exception:
                    pass

            # Health check
            try:
                async with self._session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "ok":
                            t_load1 = time.perf_counter()
                            load_ms = (t_load1 - t_load0) * 1000.0
                            mem_ready = _proc_mem_mb(pid) if pid is not None else None

                            logger.info(f"✓ Server ready at {self.base_url} (load_latency_ms={load_ms:.1f})")
                            if mem_ready:
                                logger.info(
                                    "Memory at ready: RSS=%.1fMB%s VMS=%.1fMB | Peaks during load: RSS=%.1fMB%s VMS=%.1fMB",
                                    mem_ready["rss_mb"],
                                    (f" USS={mem_ready['uss_mb']:.1f}MB," if mem_ready["uss_mb"] is not None else ","),
                                    mem_ready["vms_mb"],
                                    rss_peak,
                                    (f" USS={uss_peak:.1f}MB," if uss_peak is not None else ","),
                                    vms_peak,
                                )

                            self.last_start_stats = {
                                "ok": True,
                                "pid": pid,
                                "base_url": self.base_url,
                                "spawn_latency_ms": (t_spawn1 - t_spawn0) * 1000.0,
                                "load_latency_ms": load_ms,
                                "mem_at_ready": mem_ready,
                                "peaks_during_load": {
                                    "rss_peak_mb": rss_peak,
                                    "vms_peak_mb": vms_peak,
                                    "uss_peak_mb": uss_peak,
                                },
                                "cmd": cmd,
                            }
                            return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

            await asyncio.sleep(self.health_check_interval)

        logger.error("Server failed to become ready within timeout")
        await self.stop()
        self.last_start_stats = {
            "ok": False,
            "pid": pid,
            "error": "startup_timeout",
            "cmd": cmd,
            "peaks_during_load": {
                "rss_peak_mb": rss_peak,
                "vms_peak_mb": vms_peak,
                "uss_peak_mb": uss_peak,
            },
        }
        return False

    async def _wait_for_health(self, timeout: float) -> bool:
        """
        Wait for server to respond to health checks.

        Note: start() now implements a richer loop for profiling.
        This method is still used by callers (e.g., adapters).
        """
        await self._ensure_session()
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.is_running:
                logger.error("Server process died during startup")
                if self.process and self.process.stderr:
                    try:
                        stderr = self.process.stderr.read()
                        if stderr:
                            logger.error(f"Server stderr: {stderr}")
                    except Exception:
                        pass
                return False

            try:
                async with self._session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "ok":
                            return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

            await asyncio.sleep(self.health_check_interval)

        return False

    async def check_health(self) -> bool:
        """Check if server is healthy."""
        if not self.is_running:
            return False

        await self._ensure_session()

        try:
            async with self._session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "ok"
                return False
        except Exception:
            return False

    async def stop(self):
        """Stop llama-server process gracefully."""
        if not self.is_running:
            logger.info("Server is not running")
            await self._close_session()
            self.process = None
            return

        logger.info("Stopping llama-server...")

        try:
            self.process.terminate()

            try:
                self.process.wait(timeout=10)
                logger.info("✓ Server stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Graceful shutdown timeout, forcing kill...")
                self.process.kill()
                self.process.wait(timeout=5)
                logger.info("✓ Server forcefully stopped")

        except Exception as e:
            logger.error(f"Error stopping server: {e}")

        finally:
            self.process = None
            await self._close_session()

    async def restart(self) -> bool:
        """Restart server."""
        logger.info("Restarting server...")
        await self.stop()
        await asyncio.sleep(1)
        return await self.start()

    async def get_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Get server information and stats.
        """
        if not await self.check_health():
            return None

        await self._ensure_session()

        try:
            async with self._session.get(
                f"{self.base_url}/props",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")

        return None

    def __enter__(self):
        raise RuntimeError("Use async context manager (async with)")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# ============================================================================
# Convenience Functions
# ============================================================================

async def start_server(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    n_gpu_layers: int = 0,
    **kwargs,
) -> LlamaServerManager:
    """
    Convenience function to start a server with minimal config.
    """
    config = ServerConfig(
        model_path=model_path,
        host=host,
        port=port,
        n_gpu_layers=n_gpu_layers,
        **kwargs,
    )

    manager = LlamaServerManager(config)

    success = await manager.start()
    if not success:
        raise RuntimeError("Failed to start llama-server")

    return manager
