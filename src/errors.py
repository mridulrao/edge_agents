"""
Edge Question Generation Pipeline - Error Handling
Custom exceptions for pipeline stages
"""

from typing import Optional
from pipeline_types import PipelineStage, InferenceErrorCause


class PipelineError(Exception):
    """Base exception for pipeline errors"""
    
    def __init__(
        self, 
        message: str, 
        stage: PipelineStage,
        recoverable: bool = False
    ):
        super().__init__(message)
        self.stage = stage
        self.recoverable = recoverable
        self.message = message
    
    def __str__(self) -> str:
        recovery_status = "recoverable" if self.recoverable else "non-recoverable"
        return f"[{self.stage.value}] {self.message} ({recovery_status})"


class ValidationError(PipelineError):
    """Error during candidate validation"""
    
    def __init__(
        self,
        message: str,
        rejected_count: int,
        valid_count: int
    ):
        super().__init__(
            message=message,
            stage=PipelineStage.POST_GEN,
            recoverable=(valid_count > 0)
        )
        self.rejected_count = rejected_count
        self.valid_count = valid_count
    
    def __str__(self) -> str:
        return (f"[{self.stage.value}] {self.message} "
                f"(valid: {self.valid_count}, rejected: {self.rejected_count})")


class ChunkingError(PipelineError):
    """Error during article chunking"""
    
    def __init__(self, message: str):
        super().__init__(
            message=message,
            stage=PipelineStage.PRE_GEN,
            recoverable=False
        )


class ModelInferenceError(Exception):
    """
    Raised when model inference fails.
    
    Attributes:
        message: Human-readable error description
        cause: Categorized error type
        chunk_id: Optional chunk identifier where error occurred
    """
    
    def __init__(
        self,
        message: str,
        cause: InferenceErrorCause,
        chunk_id: Optional[int] = None
    ):
        self.message = message
        self.cause = cause
        self.chunk_id = chunk_id
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with context."""
        base = f"[{self.cause.value}] {self.message}"
        if self.chunk_id is not None:
            base += f" (chunk_id={self.chunk_id})"
        return base
    
    def is_retryable(self) -> bool:
        """
        Determine if this error is retryable.
        
        Returns:
            True if retry might succeed, False otherwise
        """
        # Timeout and unknown errors are retryable
        # OOM, parse errors, and model issues typically aren't
        return self.cause in {
            InferenceErrorCause.TIMEOUT,
            InferenceErrorCause.UNKNOWN
        }


__all__ = [
    'InferenceErrorCause',
    'ModelInferenceError',
    'PipelineError',
    'ValidationError',
    'ChunkingError'
]