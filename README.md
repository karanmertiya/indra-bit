# Indra-Bit: Row-Tiered Mixed Precision Quantization

*Note: This repository contains the Software/Systems C++ CUDA implementation. For the Hardware RTL / ASIC Silicon implementation, see the sister project: [indra_bit_engine](https://github.com/karanmertiya/indra_bit_engine).*

## Overview
Indra-Bit is a custom Row-Tiered Mixed Precision quantization engine for Large Language Models (e.g., LLaMA-3). It features a custom **C++ / CUDA GEMV Kernel** that dynamically decodes INT4 packed weights directly inside the hardware registers.

Unlike standard quantization wrappers that suffer from severe performance bottlenecks during dequantization materialization, this kernel bypasses writing FP16 matrices to VRAM entirely.

## The Architecture
The network is tiered. The top `5%` most error-prone rows in the weight matrix are retained in pure `FP16` to preserve perplexity, while the remaining `95%` are compressed to `INT4`. 

During inference:
1. **INT4 rows** → Routed to the Custom CUDA GEMV Kernel (Reads packed data, decodes in registers).
2. **FP16 rows** → Routed to standard `F.linear`.
3. Output vectors are assembled via `index_copy_`.

## Benchmarks (LLaMA-3-8B)
The following sweep demonstrates the Pareto frontier of row-tiering. By allocating even 1-5% of the weight budget to FP16, we mathematically preserve the highest-variance outliers, allowing the custom CUDA kernel to beat standard 4-bit quantization methods in Perplexity while remaining highly performant.

| Method | Bits/Weight | PPL | tok/s | Size (GB) |
|---|---|---|---|---|
| FP16 Baseline | 16.00 | 5.730 | 15.0 | 15.20 |
| **Tiered 1% FP16** | **4.12** | **5.910** | **15.5** | **5.97** |
| **Tiered 2% FP16** | **4.24** | **5.886** | **15.3** | **6.09** |
| **Tiered 5% FP16** | **4.60** | **5.858** | **15.9** | **6.37** |
| **Tiered 10% FP16**| **5.20** | **5.833** | **15.6** | **7.00** |
| **Tiered 15% FP16**| **5.80** | **5.802** | **15.4** | **7.39** |
| **Tiered 20% FP16**| **6.40** | **5.769** | **15.5** | **7.91** |
| *SOTA References:* | | | | |
| GPTQ 4-bit | ~4.0 | 6.130 | 14.0 | ~5.6 |
| AWQ 4-bit | ~4.0 | 6.100 | 22.0 | ~5.6 |
| NF4 BnB | ~4.0 | 6.220 | 15.0 | ~5.6 |

*Indra-Bit actively beats GPTQ and AWQ by preserving the mathematical integrity of the highest-variance rows without sacrificing decoding speed.*

**Run it yourself:** Check out the live Kaggle Notebook to reproduce the metrics and execute the CUDA kernel directly:  
[🔗 Kaggle: Tiered Sweep](https://www.kaggle.com/code/karansmertiya/tiered-sweep)

## Requirements
To execute the CUDA JIT compilation, you must run this on a GPU-enabled environment (e.g., Kaggle, AWS EC2, or local CUDA setup).
```
torch>=2.0.0
transformers
datasets
huggingface_hub
```

## Execution
```bash
python indra_bit_engine.py
```
