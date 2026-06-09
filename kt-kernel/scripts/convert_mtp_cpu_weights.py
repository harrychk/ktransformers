#!/usr/bin/env python3
"""Convert DeepSeek V4 MTP (NextN) layer FP4 weights to AMXINT4 for CPU MoE.

Reads MTP weights from the original FP4 checkpoint, dequantizes them to BF16,
then uses KTMoEWrapper to quantize to AMXINT4 format and appends the result
to an existing INT4 weight directory.

Usage:
  python convert_mtp_cpu_weights.py \
    --input-path /backup2/models/DeepSeek-V4-Flash \
    --output-path /backup2/DeepSeek-V4-Flash-FP8-INT4 \
    --mtp-layer-idx 43 \
    --cpuinfer-threads 64 \
    --threadpool-count 4
"""

import argparse
import gc
import glob as glob_module
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kt_kernel import KTMoEWrapper

# Reuse FP4 dequant helpers from the main conversion script
from convert_cpu_weights_ds4 import _ue8m0_to_bf16, _dequantize_fp4_blockwise

FP4_GROUP_SIZE = 32
FP4_CHUNK = 64


def find_mtp_experts(input_path):
    """Find all MTP-format experts in a model checkpoint."""
    index_file = os.path.join(input_path, "model.safetensors.index.json")
    with open(index_file) as f:
        idx = json.load(f)

    experts = set()
    file_map = {}
    for key, shard in idx["weight_map"].items():
        if key.startswith("mtp.") and ".ffn.experts." in key:
            parts = key.split(".")
            expert_id = int(parts[4])
            experts.add(expert_id)
            file_map[key] = os.path.join(input_path, shard)

    experts = sorted(experts)
    print(f"Found MTP layer: {len(experts)} experts (indices 0–{max(experts)})")
    return experts, file_map, idx


def load_mtp_tensor(file_map, mtp_idx, expert_id, proj, suffix):
    """Load a single MTP tensor.  proj ∈ {w1,w2,w3}, suffix ∈ {weight,scale}."""
    key = f"mtp.{mtp_idx}.ffn.experts.{expert_id}.{proj}.{suffix}"
    with safe_open(file_map[key], framework="pt") as f:
        return f.get_tensor(key)


def dequant_fp4_experts(fp4_list, scale_list, proj_name=""):
    """Dequantize a list of FP4 per-expert weight+scale tensors → stacked BF16."""
    chunks = []
    for i in range(0, len(fp4_list), FP4_CHUNK):
        c_fp4 = torch.stack(fp4_list[i : i + FP4_CHUNK])
        c_scale = torch.stack(scale_list[i : i + FP4_CHUNK])
        C, N, Kp = c_fp4.shape

        flat_fp4 = c_fp4.reshape(C * N, Kp).contiguous()
        flat_scale = c_scale.reshape(C * N, c_scale.shape[2]).contiguous()
        del c_fp4, c_scale

        scale_bf16 = _ue8m0_to_bf16(flat_scale)
        if torch.cuda.is_available():
            flat_fp4 = flat_fp4.cuda()
            scale_bf16 = scale_bf16.cuda()

        bf16 = _dequantize_fp4_blockwise(flat_fp4, scale_bf16, group_size=FP4_GROUP_SIZE)
        del flat_fp4, scale_bf16

        chunks.append(bf16.cpu().reshape(C, N, Kp * 2))
        del bf16

    result = torch.cat(chunks, dim=0).contiguous()
    print(f"    {proj_name}: {len(fp4_list)} experts → {list(result.shape)}")
    return result


def _load_binary_tensor(filepath):
    """Load a raw binary tensor file saved by KTMoEWrapper."""
    with open(filepath, "rb") as f:
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def load_kt_tensors_from_disk(scratch_path, layer_idx, num_experts, threadpool_count):
    """Read back INT4 .kt files → dict of safetensor-ready tensors.

    Mirrors OnlineQuantConverter._load_layer_tensors_from_disk.
    """
    layer_path = os.path.join(scratch_path, f"_layer_{layer_idx}")
    if not os.path.exists(layer_path):
        raise FileNotFoundError(f"Layer folder not found: {layer_path}")

    tensors = {}
    for numa_idx in range(threadpool_count):
        numa_folder = os.path.join(layer_path, f"_numa_{numa_idx}")
        if not os.path.exists(numa_folder):
            print(f"  Warning: NUMA folder not found: {numa_folder}")
            continue

        for expert_id in range(num_experts):
            for proj_tag, proj_key in [("down", "ffn_down_exps"),
                                        ("gate", "ffn_gate_exps"),
                                        ("up", "ffn_up_exps")]:
                q_pat = os.path.join(numa_folder, f"INT4_{proj_tag}_{expert_id}_*Byte_quant_.kt")
                s_pat = os.path.join(numa_folder, f"INT4_{proj_tag}_{expert_id}_*Byte_scale_.kt")

                q_files = glob_module.glob(q_pat)
                s_files = glob_module.glob(s_pat)

                w_key = f"blk.{layer_idx}.{proj_key}.{expert_id}.numa.{numa_idx}.weight"
                s_key = f"blk.{layer_idx}.{proj_key}.{expert_id}.numa.{numa_idx}.scale"

                if q_files:
                    if len(q_files) > 1:
                        raise RuntimeError(f"Multiple quant files: {q_files}")
                    tensors[w_key] = _load_binary_tensor(q_files[0])
                if s_files:
                    if len(s_files) > 1:
                        raise RuntimeError(f"Multiple scale files: {s_files}")
                    tensors[s_key] = _load_binary_tensor(s_files[0])

    return tensors


def main():
    p = argparse.ArgumentParser(description="Convert MTP layer FP4→AMXINT4")
    p.add_argument("--input-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--mtp-layer-idx", type=int, required=True)
    p.add_argument("--cpuinfer-threads", type=int, default=64)
    p.add_argument("--threadpool-count", type=int, default=4)
    p.add_argument("--num-experts-per-tok", type=int, default=8)
    args = p.parse_args()

    out_idx = args.mtp_layer_idx  # e.g. 43 = num_hidden_layers

    # ── 1. Discover MTP experts ────────────────────────────────────
    experts, file_map, _ = find_mtp_experts(args.input_path)
    E = len(experts)

    # Infer dimensions from first expert
    w1 = load_mtp_tensor(file_map, 0, experts[0], "w1", "weight").view(torch.uint8)
    N1, Kp = w1.shape                     # N1=2*I, Kp=H/2
    hidden_size = Kp * 2
    moe_intermediate_size = N1            # 2 × full intermediate_size
    del w1

    w2 = load_mtp_tensor(file_map, 0, experts[0], "w2", "weight").view(torch.uint8)
    print(f"Dims: hidden={hidden_size}  moe_intermediate={moe_intermediate_size}  "
          f"w2_shape={list(w2.shape)}")
    del w2

    # ── 2. Load & dequantize all experts ───────────────────────────
    print(f"Loading {E} experts …")
    t0 = time.time()

    gate_w, gate_s = [], []
    up_w,   up_s   = [], []
    down_w, down_s = [], []

    for eid in experts:
        for lst_w, lst_s, proj in [(gate_w, gate_s, "w1"),
                                    (up_w,   up_s,   "w3"),
                                    (down_w, down_s, "w2")]:
            w = load_mtp_tensor(file_map, 0, eid, proj, "weight")
            s = load_mtp_tensor(file_map, 0, eid, proj, "scale")
            lst_w.append(w.view(torch.uint8) if w.dtype != torch.uint8 else w)
            lst_s.append(s)

    print(f"  loaded in {time.time()-t0:.1f}s")

    gate_bf16 = dequant_fp4_experts(gate_w, gate_s, "gate(w1)")
    del gate_w, gate_s; gc.collect()
    up_bf16   = dequant_fp4_experts(up_w,   up_s,   "up(w3)")
    del up_w,   up_s;   gc.collect()
    down_bf16 = dequant_fp4_experts(down_w, down_s, "down(w2)")
    del down_w, down_s; gc.collect()

    # ── 3. Quantize via KTMoEWrapper ──────────────────────────────
    scratch_dir = os.path.join(args.output_path, f"_scratch_mtp_{out_idx}")
    os.makedirs(scratch_dir, exist_ok=True)

    gpu_mask = torch.zeros(E, dtype=torch.bool)
    phys2log = torch.arange(E, dtype=torch.int64)

    print("Quantizing to AMXINT4 …")
    wrapper = KTMoEWrapper(
        layer_idx=out_idx,
        num_experts=E,
        num_experts_per_tok=args.num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=args.threadpool_count,
        weight_path=scratch_dir,
        chunked_prefill_size=512,
        cpu_save=True,
        method="AMXINT4",
    )
    wrapper.load_weights_from_tensors(gate_bf16, up_bf16, down_bf16, phys2log)
    del wrapper, gate_bf16, up_bf16, down_bf16; gc.collect()

    # ── 4. Read back quantized weights ────────────────────────────
    print("Reading back .kt files …")
    new_tensors = load_kt_tensors_from_disk(
        scratch_dir, out_idx, E, args.threadpool_count,
    )
    print(f"  → {len(new_tensors)} tensors")

    # ── 5. Write shard & update index ─────────────────────────────
    existing_shards = sorted(glob_module.glob(
        os.path.join(args.output_path, "model-*-of-*.safetensors")))
    max_n = 0
    old_total = 0
    for s in existing_shards:
        base = os.path.basename(s)
        parts = base.split("-")
        try:
            max_n = max(max_n, int(parts[1]))
            old_total = max(old_total, int(parts[2].split("of")[1].replace(".safetensors", "")))
        except (ValueError, IndexError):
            pass

    new_n = max_n + 1
    new_total = old_total + 1

    # Rename existing shards to new total
    for s in existing_shards:
        base = os.path.basename(s)
        new_base = base.replace(f"-of-{old_total:05d}", f"-of-{new_total:05d}")
        if new_base != base:
            os.rename(s, os.path.join(args.output_path, new_base))

    # Write MTP shard
    shard_name = f"model-{new_n:05d}-of-{new_total:05d}.safetensors"
    shard_path = os.path.join(args.output_path, shard_name)
    # Save in chunks to avoid huge single write
    tensor_items = list(new_tensors.items())
    chunk = {}
    for k, v in tensor_items:
        chunk[k] = v
    save_file(chunk, shard_path)
    print(f"  Wrote {len(new_tensors)} tensors → {shard_name}")

    # Update index.json
    index_path = os.path.join(args.output_path, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)

    for key in new_tensors:
        idx["weight_map"][key] = shard_name

    for key, shard in idx["weight_map"].items():
        if f"-of-{old_total:05d}" in shard:
            idx["weight_map"][key] = shard.replace(
                f"-of-{old_total:05d}", f"-of-{new_total:05d}")

    with open(index_path, "w") as f:
        json.dump(idx, f, indent=2)
    print(f"  Updated index.json")

    # Clean up scratch
    shutil.rmtree(scratch_dir, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
