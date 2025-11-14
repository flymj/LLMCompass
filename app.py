"""Streamlit dashboard for exploring LLMCompass hardware/software combinations."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import plotly.express as px
import streamlit as st

from hardware_model.compute_module import ComputeModule, compute_module_dict
from hardware_model.device import Device, device_dict
from hardware_model.interconnect import (
    InterConnectModule,
    LinkModule,
    TopologyType,
    interconnect_module_dict,
)
from hardware_model.io_module import IO_module_dict
from hardware_model.memory_module import MemoryModule, memory_module_dict
from hardware_model.system import System
from software_model.transformer import (
    TransformerBlockAutoRegressionTP,
    TransformerBlockInitComputationTP,
)
from software_model.utils import Tensor, data_type_dict

st.set_page_config(page_title="LLMCompass Explorer", layout="wide")

TRANSFORMER_STAGE_LABELS = [
    "Q/K/V projections",
    "Q × Kᵀ",
    "Attention × V",
    "Output projection",
    "FFN expand",
    "FFN project",
    "Softmax",
    "LayerNorm (post-attn)",
    "LayerNorm (post-ffn)",
    "GeLU",
    "AllReduce (attention)",
    "AllReduce (ffn)",
    "FlashAttention-3",
]

ATTENTION_KERNEL_OPTIONS = {
    "Standard matmul + softmax": "standard",
    "FlashAttention-3": "flash-attention-3",
}


def resolve_module_name(target, registry: Dict[str, object]) -> str:
    for name, module in registry.items():
        if module is target:
            return name
    return "custom"


def describe_compute_module(module: ComputeModule) -> Dict[str, float]:
    return {
        "cores": module.core_count,
        "clock_GHz": round(module.clock_freq / 1e9, 2),
        "l2_MB": round(module.l2_size / 1024**2, 1),
        "vector_TFLOPS": round(module.total_vector_flops / 1e12, 2),
        "tensor_TFLOPS": round(module.total_systolic_array_flops / 1e12, 2),
    }


def describe_memory_module(module: MemoryModule) -> Dict[str, float | str | None]:
    capacity_gb = module.memory_capacity / 1e9 if module.memory_capacity else None
    bandwidth_tb = (
        module.bandwidth_byte_per_sec / 1e12
        if module.bandwidth_byte_per_sec is not None
        else None
    )
    return {
        "type": module.memory_type,
        "capacity_GB": None if capacity_gb is None else round(capacity_gb, 2),
        "bandwidth_TBps": None if bandwidth_tb is None else round(bandwidth_tb, 2),
    }


def parse_transformer_breakdown(log: str | None) -> List[Tuple[str, float]]:
    if not log:
        return []
    try:
        values = [float(item.strip()) for item in log.split(",")]
    except ValueError:
        return []
    labels = TRANSFORMER_STAGE_LABELS[: len(values)]
    return list(zip(labels, values))


def parse_int_list(raw: str, fallback: Sequence[int]) -> List[int]:
    cleaned = [token.strip() for token in raw.split(",") if token.strip()]
    if not cleaned:
        return list(fallback)
    values: List[int] = []
    for token in cleaned:
        try:
            values.append(int(token))
        except ValueError:
            st.warning(f"Ignoring invalid value '{token}'.")
    return values or list(fallback)


def build_system(device: Device, interconnect: InterConnectModule) -> System:
    return System(device, interconnect)


def render_breakdown_chart(
    breakdown: List[Tuple[str, float]], *, chart_key: str
) -> None:
    if not breakdown:
        return
    names = [name for name, _ in breakdown]
    latencies_ms = [value * 1e3 for _, value in breakdown]
    fig = px.bar(
        x=names,
        y=latencies_ms,
        labels={"x": "Stage", "y": "Latency (ms)"},
        title="Roofline latency breakdown",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def summarize_hardware(device: Device) -> None:
    compute_summary = describe_compute_module(device.compute_module)
    memory_summary = describe_memory_module(device.memory_module)
    io_bandwidth_tb = device.io_module.bandwidth / 1e12
    col1, col2, col3 = st.columns(3)
    col1.metric("Cores", compute_summary["cores"])
    col2.metric("Clock (GHz)", compute_summary["clock_GHz"])
    col3.metric("L2 cache (MB)", compute_summary["l2_MB"])
    col4, col5, col6 = st.columns(3)
    col4.metric("Vector throughput (TFLOPS)", compute_summary["vector_TFLOPS"])
    col5.metric("Tensor throughput (TFLOPS)", compute_summary["tensor_TFLOPS"])
    col6.metric("IO bandwidth (TB/s)", round(io_bandwidth_tb, 3))
    st.caption(
        f"Memory: {memory_summary['capacity_GB']} GB {memory_summary['type']}"
        f" @ {memory_summary['bandwidth_TBps']} TB/s"
    )


def run_transformer_prefill(
    *,
    d_model: int,
    n_heads: int,
    device_count: int,
    batch_size: int,
    seq_len: int,
    data_type_key: str,
    system: System,
    attention_kernel: str,
) -> Tuple[float, List[Tuple[str, float]]]:
    data_type = data_type_dict[data_type_key]
    model = TransformerBlockInitComputationTP(
        d_model=d_model,
        n_heads=n_heads,
        device_count=device_count,
        data_type=data_type,
        attention_kernel=attention_kernel,
    )
    _ = model(Tensor([batch_size, seq_len, d_model], data_type))
    latency = model.roofline_model(system)
    breakdown = parse_transformer_breakdown(getattr(model, "roofline_log", None))
    return latency, breakdown


def run_transformer_decode(
    *,
    d_model: int,
    n_heads: int,
    device_count: int,
    batch_size: int,
    kv_seq_len: int,
    data_type_key: str,
    system: System,
    attention_kernel: str,
) -> Tuple[float, List[Tuple[str, float]]]:
    data_type = data_type_dict[data_type_key]
    model = TransformerBlockAutoRegressionTP(
        d_model=d_model,
        n_heads=n_heads,
        device_count=device_count,
        data_type=data_type,
        attention_kernel=attention_kernel,
    )
    _ = model(Tensor([batch_size, 1, d_model], data_type), kv_seq_len)
    latency = model.roofline_model(system)
    breakdown = parse_transformer_breakdown(getattr(model, "roofline_log", None))
    return latency, breakdown


st.title("LLMCompass interactive explorer")
st.write(
    "Configure hardware, IO, and software workloads to inspect roofline latencies."
)

# Sidebar hardware configuration
st.sidebar.header("Hardware stack")
hardware_mode = st.sidebar.selectbox(
    "Select hardware source",
    ("Registered device", "Custom modules"),
)

if hardware_mode == "Registered device":
    preset_name = st.sidebar.selectbox(
        "Device preset", sorted(device_dict.keys()), index=0
    )
    current_device = device_dict[preset_name]
    compute_name = resolve_module_name(
        current_device.compute_module, compute_module_dict
    )
    io_name = resolve_module_name(current_device.io_module, IO_module_dict)
    memory_name = resolve_module_name(
        current_device.memory_module, memory_module_dict
    )
else:
    preset_name = "custom"
    compute_name = st.sidebar.selectbox(
        "Compute module", sorted(compute_module_dict.keys())
    )
    io_name = st.sidebar.selectbox("IO module", sorted(IO_module_dict.keys()))
    memory_name = st.sidebar.selectbox(
        "Memory module", sorted(memory_module_dict.keys())
    )
    current_device = Device(
        compute_module_dict[compute_name],
        IO_module_dict[io_name],
        memory_module_dict[memory_name],
    )

st.sidebar.markdown(
    f"**Compute:** `{compute_name}`  \n"
    f"**IO:** `{io_name}`  \n"
    f"**Memory:** `{memory_name}`"
)

st.sidebar.header("Interconnect")
interconnect_mode = st.sidebar.selectbox(
    "Select interconnect source", ("Preset", "Custom")
)
if interconnect_mode == "Preset":
    interconnect_name = st.sidebar.selectbox(
        "Preset", sorted(interconnect_module_dict.keys())
    )
    interconnect = interconnect_module_dict[interconnect_name]
else:
    topology_label = st.sidebar.selectbox("Topology", ("FC", "RING"))
    device_count_ic = st.sidebar.number_input(
        "Device count", min_value=1, value=4, step=1
    )
    link_count = st.sidebar.number_input(
        "Links per device", min_value=1, value=12, step=1
    )
    bw_per_dir_gbps = st.sidebar.number_input(
        "Bandwidth per direction (GB/s)", min_value=1.0, value=25.0, step=1.0
    )
    aggregate_bw_gbps = st.sidebar.number_input(
        "Bidirectional bandwidth (GB/s)", min_value=1.0, value=50.0, step=1.0
    )
    latency_ns = st.sidebar.number_input(
        "Link latency (ns)", min_value=1.0, value=100.0, step=1.0
    )
    flit_size = st.sidebar.number_input("Flit size (B)", min_value=1, value=16, step=1)
    payload = st.sidebar.number_input(
        "Max payload (B)", min_value=16, value=256, step=16
    )
    header = st.sidebar.number_input(
        "Header size (B)", min_value=4, value=16, step=4
    )
    link = LinkModule(
        bandwidth_per_direction=bw_per_dir_gbps * 1e9,
        bandwidth_both_direction=aggregate_bw_gbps * 1e9,
        latency=latency_ns * 1e-9,
        flit_size=flit_size,
        max_payload_size=payload,
        header_size=header,
    )
    interconnect = InterConnectModule(
        device_count=int(device_count_ic),
        topology=TopologyType[topology_label],
        link_module=link,
        link_count_per_device=int(link_count),
    )
    interconnect_name = f"custom_{topology_label.lower()}"

system = build_system(current_device, interconnect)

st.subheader("Selected hardware summary")
summarize_hardware(current_device)

st.header("Software workload")
software_choice = st.selectbox(
    "Select workload",
    ("Transformer (prefill)", "Transformer (decode)")
)

data_type_options = sorted(data_type_dict.keys())
default_dtype_index = (
    data_type_options.index("fp16") if "fp16" in data_type_options else 0
)
data_type_choice = st.selectbox(
    "Numerical precision", data_type_options, index=default_dtype_index
)
attention_choice_label = st.selectbox(
    "Attention kernel", list(ATTENTION_KERNEL_OPTIONS.keys())
)
attention_kernel = ATTENTION_KERNEL_OPTIONS[attention_choice_label]

def validate_transformer_inputs(
    *, d_model: int, n_heads: int, device_count: int
) -> List[str]:
    errors: List[str] = []
    if d_model % n_heads != 0:
        errors.append("d_model must be divisible by n_heads.")
    if d_model % device_count != 0:
        errors.append("d_model must be divisible by the tensor parallel device count.")
    if n_heads % device_count != 0:
        errors.append("n_heads must be divisible by the tensor parallel device count.")
    return errors


if software_choice == "Transformer (prefill)":
    d_model = st.number_input("Hidden size (d_model)", 128, 262144, 12288, 128)
    n_heads = st.number_input("Attention heads", 1, 1024, 96, 1)
    tp_devices = st.number_input(
        "Tensor parallel device count", 1, 128, 4, 1
    )
    batch_size = st.number_input("Batch size", 1, 4096, 8, 1)
    seq_len = st.number_input("Sequence length", 1, 65536, 2048, 1)
    sweep_param = st.selectbox(
        "Parameter to explore", ("None", "Batch size", "Sequence length")
    )
    default_values = str(batch_size if sweep_param == "Batch size" else seq_len)
    sweep_values_raw = st.text_input(
        "Comma-separated values", value=default_values
    )
    errors = validate_transformer_inputs(
        d_model=d_model, n_heads=n_heads, device_count=tp_devices
    )
    if errors:
        for msg in errors:
            st.error(msg)
    run_prefill = st.button("Run roofline analysis", type="primary")

    if run_prefill and not errors:
        sweep_values = parse_int_list(
            sweep_values_raw,
            [batch_size] if sweep_param == "Batch size" else [seq_len],
        )
        results = []
        for value in sweep_values:
            current_batch = value if sweep_param == "Batch size" else batch_size
            current_seq = value if sweep_param == "Sequence length" else seq_len
                latency, breakdown = run_transformer_prefill(
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    device_count=int(tp_devices),
                    batch_size=int(current_batch),
                    seq_len=int(current_seq),
                    data_type_key=data_type_choice,
                    system=system,
                    attention_kernel=attention_kernel,
                )
            results.append(
                {
                    "batch_size": current_batch,
                    "seq_len": current_seq,
                    "latency_s": latency,
                    "breakdown": breakdown,
                }
            )
        st.success(
            f"Latest configuration latency: {results[-1]['latency_s'] * 1e3:.3f} ms"
        )
        breakdown = results[-1]["breakdown"]
        if breakdown:
            st.subheader("Latency breakdown")
            st.dataframe(
                [
                    {
                        "Stage": name,
                        "Latency (ms)": round(value * 1e3, 4),
                    }
                    for name, value in breakdown
                ],
                use_container_width=True,
                hide_index=True,
            )
            render_breakdown_chart(breakdown, chart_key="prefill_breakdown")
        if len(results) > 1:
            sweep_axis = "Batch size" if sweep_param == "Batch size" else "Sequence length"
            fig = px.line(
                x=[r["batch_size"] if sweep_param == "Batch size" else r["seq_len"] for r in results],
                y=[r["latency_s"] * 1e3 for r in results],
                markers=True,
                labels={"x": sweep_axis, "y": "Latency (ms)"},
                title=f"Latency vs. {sweep_axis}",
            )
            st.plotly_chart(fig, use_container_width=True, key="prefill_sweep")

else:
    d_model = st.number_input("Hidden size (d_model)", 128, 262144, 12288, 128)
    n_heads = st.number_input("Attention heads", 1, 1024, 96, 1)
    tp_devices = st.number_input(
        "Tensor parallel device count", 1, 128, 4, 1
    )
    batch_size = st.number_input("Batch size", 1, 4096, 8, 1)
    kv_seq_len = st.number_input(
        "KV cache sequence length", 1, 65536, 2048, 1
    )
    sweep_param = st.selectbox(
        "Parameter to explore", ("None", "Batch size", "KV cache length")
    )
    default_values = str(batch_size if sweep_param == "Batch size" else kv_seq_len)
    sweep_values_raw = st.text_input(
        "Comma-separated values", value=default_values
    )
    errors = validate_transformer_inputs(
        d_model=d_model, n_heads=n_heads, device_count=tp_devices
    )
    if errors:
        for msg in errors:
            st.error(msg)
    run_decode = st.button("Run roofline analysis", type="primary")

    if run_decode and not errors:
        sweep_values = parse_int_list(
            sweep_values_raw,
            [batch_size] if sweep_param == "Batch size" else [kv_seq_len],
        )
        results = []
        for value in sweep_values:
            current_batch = value if sweep_param == "Batch size" else batch_size
            current_kv = value if sweep_param == "KV cache length" else kv_seq_len
            latency, breakdown = run_transformer_decode(
                d_model=int(d_model),
                n_heads=int(n_heads),
                device_count=int(tp_devices),
                batch_size=int(current_batch),
                kv_seq_len=int(current_kv),
                data_type_key=data_type_choice,
                system=system,
                attention_kernel=attention_kernel,
            )
            results.append(
                {
                    "batch_size": current_batch,
                    "kv_seq_len": current_kv,
                    "latency_s": latency,
                    "breakdown": breakdown,
                }
            )
        st.success(
            f"Latest configuration latency: {results[-1]['latency_s'] * 1e3:.3f} ms"
        )
        breakdown = results[-1]["breakdown"]
        if breakdown:
            st.subheader("Latency breakdown")
            st.dataframe(
                [
                    {
                        "Stage": name,
                        "Latency (ms)": round(value * 1e3, 4),
                    }
                    for name, value in breakdown
                ],
                use_container_width=True,
                hide_index=True,
            )
            render_breakdown_chart(breakdown, chart_key="decode_breakdown")
        if len(results) > 1:
            sweep_axis = "Batch size" if sweep_param == "Batch size" else "KV cache length"
            fig = px.line(
                x=[
                    r["batch_size"]
                    if sweep_param == "Batch size"
                    else r["kv_seq_len"]
                    for r in results
                ],
                y=[r["latency_s"] * 1e3 for r in results],
                markers=True,
                labels={"x": sweep_axis, "y": "Latency (ms)"},
                title=f"Latency vs. {sweep_axis}",
            )
            st.plotly_chart(fig, use_container_width=True, key="decode_sweep")
