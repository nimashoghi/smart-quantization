from pytorch_lightning.profiler.base import BaseProfiler
from smart_compress.compress.base import CompressionAlgorithmBase


class Globals:
    compression: CompressionAlgorithmBase = None
    profiler: BaseProfiler = None
