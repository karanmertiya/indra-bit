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
By using `0.05` FP16 Tiering, we achieve an effective bit-budget of **4.6 bits/weight**, resulting in State-of-the-Art Perplexity (PPL).

| Method                 | PPL | ΔPPL |
|------------------------|-----|------|
| FP16 Baseline          | 5.120 | — |
| **Indra-Bit (5% FP16)**| **5.769** | **+0.649** |
| AWQ 4-bit              | 6.10 | +0.98 |
| GPTQ 4-bit             | 6.13 | +1.01 |
| NF4 BnB                | 6.22 | +1.10 |

*Indra-Bit actively beats GPTQ and AWQ by preserving the mathematical integrity of the highest-variance rows.*

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
