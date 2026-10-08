"""Register official Jina architectures in spawned vLLM workers."""
from deepseek_ocr_mtp import register

register()


class JinaRegistration:
    """Import-only worker extension; no inference overrides."""
