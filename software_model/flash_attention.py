"""FlashAttention-3 operator modeling."""
from __future__ import annotations

from dataclasses import dataclass

from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import DataType, Tensor


@dataclass
class _FlashAttentionShape:
    batch: int
    heads: int
    query_len: int
    dim_per_head: int
    kv_len: int

    @property
    def output_shape(self) -> list[int]:
        return [self.batch, self.heads, self.query_len, self.dim_per_head]


class FlashAttention3(Operator):
    """A simplified FlashAttention v3 roofline model.

    The operator fuses the QK^T matmul, softmax, and attention-value matmul.
    The implementation intentionally mirrors the Matmul operator API so it can
    be swapped into existing transformer graphs without affecting tensor shapes.
    """

    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape: _FlashAttentionShape | None = None
        self.q_size = 0
        self.k_size = 0
        self.v_size = 0
        self.output_size = 0

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        assert q.data_type == self.data_type
        assert k.data_type == self.data_type
        assert v.data_type == self.data_type
        assert len(q.shape) == 4, "FlashAttention expects [B, H, Lq, Dh] tensors"
        assert len(k.shape) == 4 and len(v.shape) == 4
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[1] == k.shape[1] == v.shape[1]
        assert q.shape[3] == k.shape[2]
        assert v.shape[2] == k.shape[3]
        assert v.shape[3] == q.shape[3]

        self.shape = _FlashAttentionShape(
            batch=q.shape[0],
            heads=q.shape[1],
            query_len=q.shape[2],
            dim_per_head=q.shape[3],
            kv_len=k.shape[3],
        )
        output = Tensor(self.shape.output_shape, self.data_type)
        self.q_size = q.size
        self.k_size = k.size
        self.v_size = v.size
        self.output_size = output.size
        self.flop_count = self._qk_flops + self._av_flops + self._softmax_flops()
        self.load_count = self.q_size + self.k_size + self.v_size
        self.store_count = self.output_size
        self.io_count = self.load_count + self.store_count
        self.peak_memory_usage = (
            self.q_size + self.k_size + self.v_size + self.output_size
        )
        return output

    @property
    def _qk_flops(self) -> int:
        assert self.shape is not None
        return (
            2
            * self.shape.batch
            * self.shape.heads
            * self.shape.query_len
            * self.shape.dim_per_head
            * self.shape.kv_len
        )

    @property
    def _av_flops(self) -> int:
        assert self.shape is not None
        return (
            2
            * self.shape.batch
            * self.shape.heads
            * self.shape.query_len
            * self.shape.kv_len
            * self.shape.dim_per_head
        )

    def _softmax_flops(self) -> int:
        assert self.shape is not None
        return (
            self.shape.batch
            * self.shape.heads
            * self.shape.query_len
            * self.shape.kv_len
            * 10
        )

    def roofline_model(self, pcb_module: Device):
        assert self.shape is not None, "Call the operator before profiling."
        qk_latency = self._qk_flops / pcb_module.compute_module.total_systolic_array_flops
        av_latency = self._av_flops / pcb_module.compute_module.total_systolic_array_flops
        softmax_latency = (
            self._softmax_flops() / pcb_module.compute_module.total_vector_flops
        )
        compute_latency = qk_latency + softmax_latency + av_latency
        bandwidth = min(
            pcb_module.io_module.bandwidth,
            pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq,
        )
        io_bytes = self.io_count * self.data_type.word_size
        io_latency = io_bytes / bandwidth
        self.roofline_latency = max(compute_latency, io_latency)
        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode=None):
        # FlashAttention is modeled analytically, so reuse the roofline result.
        return self.roofline_model(pcb_module)
