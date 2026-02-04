"""
Edge Question Generation Pipeline - LlamaServer Manager
Manages llama-server process lifecycle and health monitoring
"""

import asyncio
import subprocess
import time
import signal
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for llama-server instance"""
    
    # Model settings
    model_path: str
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080
    
    # Context settings
    ctx_size: int = 4096
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
        health_check_interval: float = 1.0
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
    
    @property
    def base_url(self) -> str:
        """Get server base URL"""
        return f"http://{self.config.host}:{self.config.port}"
    
    @property
    def is_running(self) -> bool:
        """Check if process is running"""
        return self.process is not None and self.process.poll() is None
    
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
        
        if not model_path.suffix == ".gguf":
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
                    text=True
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
                f"No working llama binary found. Tried:\n" +
                "\n".join(f"  - {b}" for b in binaries_to_try) +
                f"\n\nTo install llama.cpp:\n"
                f"  git clone https://github.com/ggerganov/llama.cpp.git\n"
                f"  cd llama.cpp\n"
                f"  mkdir build && cd build\n"
                f"  cmake .. -DGGML_METAL=ON  # For macOS\n"
                f"  cmake --build . --config Release\n"
                f"  # Binary will be in build/bin/llama-server or build/bin/llama-cli"
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
        
        # Build command
        cmd = self.config.to_cmd_args(self.llama_server_path, self._is_llama_cli)
        
        logger.info(f"Starting {'llama-cli' if self._is_llama_cli else 'llama-server'}: {' '.join(cmd)}")
        
        # Start process
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
        except Exception as e:
            logger.error(f"Failed to start server process: {e}")
            return False
        
        # Wait for server to become ready
        logger.info(f"Waiting for server to become ready (timeout: {self.startup_timeout}s)...")
        
        ready = await self._wait_for_health(self.startup_timeout)
        
        if ready:
            logger.info(f"âœ“ Server ready at {self.base_url}")
            return True
        else:
            logger.error("Server failed to become ready within timeout")
            await self.stop()
            return False
    
    async def _wait_for_health(self, timeout: float) -> bool:
        """
        Wait for server to respond to health checks.
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if server became healthy, False otherwise
        """
        await self._ensure_session()
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if process died
            if not self.is_running:
                logger.error("Server process died during startup")
                # Log stderr
                if self.process and self.process.stderr:
                    stderr = self.process.stderr.read()
                    if stderr:
                        logger.error(f"Server stderr: {stderr}")
                return False
            
            # Check health endpoint
            try:
                async with self._session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "ok":
                            return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                # Server not ready yet, continue waiting
                pass
            
            await asyncio.sleep(self.health_check_interval)
        
        return False
    
    async def check_health(self) -> bool:
        """
        Check if server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        if not self.is_running:
            return False
        
        await self._ensure_session()
        
        try:
            async with self._session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "ok"
                return False
        except Exception:
            return False
    
    async def stop(self):
        """Stop llama-server process gracefully"""
        if not self.is_running:
            logger.info("Server is not running")
            return
        
        logger.info("Stopping llama-server...")
        
        # Try graceful shutdown first (SIGTERM)
        try:
            self.process.terminate()
            
            # Wait up to 10 seconds for graceful shutdown
            try:
                self.process.wait(timeout=10)
                logger.info("âœ“ Server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning("Graceful shutdown timeout, forcing kill...")
                self.process.kill()
                self.process.wait(timeout=5)
                logger.info("âœ“ Server forcefully stopped")
        
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
        
        finally:
            self.process = None
            await self._close_session()
    
    async def restart(self) -> bool:
        """
        Restart server.
        
        Returns:
            True if restart successful, False otherwise
        """
        logger.info("Restarting server...")
        await self.stop()
        await asyncio.sleep(1)  # Brief pause
        return await self.start()
    
    async def get_server_info(self) -> Optional[Dict[str, Any]]:
        """
        Get server information and stats.
        
        Returns:
            Server info dict or None if unavailable
        """
        if not await self.check_health():
            return None
        
        await self._ensure_session()
        
        try:
            async with self._session.get(
                f"{self.base_url}/props",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
        
        return None
    
    def __enter__(self):
        """Context manager support (sync)"""
        raise RuntimeError("Use async context manager (async with)")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


# ============================================================================
# Convenience Functions
# ============================================================================

async def start_server(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    n_gpu_layers: int = 0,
    **kwargs
) -> LlamaServerManager:
    """
    Convenience function to start a server with minimal config.
    
    Args:
        model_path: Path to GGUF model file
        host: Server host
        port: Server port
        n_gpu_layers: Number of GPU layers (0=CPU, -1=all)
        **kwargs: Additional ServerConfig parameters
        
    Returns:
        Started LlamaServerManager instance
    """
    config = ServerConfig(
        model_path=model_path,
        host=host,
        port=port,
        n_gpu_layers=n_gpu_layers,
        **kwargs
    )
    
    manager = LlamaServerManager(config)
    
    success = await manager.start()
    if not success:
        raise RuntimeError("Failed to start llama-server")
    
    return manager


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of LlamaServerManager"""
    
    # Option 1: Using context manager (recommended)
    config = ServerConfig(
        model_path="models/mistral-7b-instruct-Q4_K_M.gguf",
        host="127.0.0.1",
        port=8080,
        ctx_size=4096,
        n_gpu_layers=35,  # Use GPU if available
        n_threads=8
    )
    
    async with LlamaServerManager(config) as server:
        print(f"Server running at {server.base_url}")
        
        # Get server info
        info = await server.get_server_info()
        if info:
            print(f"Server info: {info}")
        
        # Do work here...
        # Your pipeline can now connect to the server
        
        await asyncio.sleep(5)  # Simulate work
    
    print("Server stopped")
    
    # Option 2: Manual management
    manager = LlamaServerManager(config)
    
    if await manager.start():
        print("Server started successfully")
        
        # Check health
        healthy = await manager.check_health()
        print(f"Server healthy: {healthy}")
        
        # ... do work ...
        
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())