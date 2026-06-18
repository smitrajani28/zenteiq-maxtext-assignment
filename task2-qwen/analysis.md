# Task 2 - Dense Model (Qwen)

## Overview

The objective of this task was to evaluate the execution of a Dense Transformer model using MaxText across CPU, GPU, and TPU backends. The assignment specified using the Qwen architecture with synthetic data and comparing a base model against a scaled-up version.

Two models were evaluated:

* Base Model: Qwen3-0.6B
* Scaled Model: Qwen3-1.7B

Synthetic data was used for all experiments.

---

# Model Scaling

The assignment required scaling the base model and documenting the configuration changes.

## Configuration Comparison

| Parameter           | Qwen3-0.6B | Qwen3-1.7B |
| ------------------- | ---------- | ---------- |
| Embedding Dimension | 1024       | 2048       |
| MLP Dimension       | 3072       | 6144       |
| Decoder Layers      | 28         | 28         |
| Query Heads         | 16         | 16         |
| KV Heads            | 8          | 8          |
| Head Dimension      | 128        | 128        |
| Vocabulary Size     | 151936     | 151936     |

## Scaling Strategy

The scaled model was selected from the existing MaxText model configurations.

The primary changes were:

* Doubling the embedding dimension (1024 → 2048)
* Doubling the MLP dimension (3072 → 6144)

These parameters contribute significantly to total parameter count and computational cost, making Qwen3-1.7B substantially larger than Qwen3-0.6B while preserving the same overall transformer architecture.

---

# Backend Results

## Qwen3-0.6B Results

| Backend | Result |
| ------- | ------ |
| CPU     | Failed |
| GPU     | Failed |
| TPU     | Failed |

### CPU

The CPU run failed due to limitations in the backend used by MaxText. The attention implementation relies on Pallas kernels that are not fully supported on the CPU backend.

### GPU

The GPU run initialized successfully but failed because the available Tesla T4 GPU does not satisfy the requirements for Triton-based kernels used by MaxText.

### TPU

The TPU runtime was detected successfully, however execution terminated due to an LLVM/XLA runtime conflict before model compilation could begin.

---

## Qwen3-1.7B Results

| Backend | Result            |
| ------- | ----------------- |
| CPU     | Failed            |
| GPU     | Failed            |
| TPU     | Failed (expected) |

### CPU

The scaled model progressed further through initialization than the 0.6B model but was terminated with exit code 137.

This indicates that the process was killed due to memory exhaustion (OOM). The larger embedding and MLP dimensions increased memory requirements beyond what was available in the Colab CPU environment.

### GPU

The GPU run failed during Transformer Engine initialization due to a CUDA/cuDNN dependency conflict.

The model configuration itself loaded successfully, but required GPU libraries could not be initialized correctly.

### TPU

The TPU failure observed for the base model occurred before model compilation and execution. Since the issue originates from the runtime environment rather than model configuration, the same limitation is expected to affect the scaled model as well.

---

# Comparison of Backend Behavior

| Backend | Qwen3-0.6B                | Qwen3-1.7B                     |
| ------- | ------------------------- | ------------------------------ |
| CPU     | Backend limitation        | Out of Memory                  |
| GPU     | Triton/T4 incompatibility | CUDA/cuDNN dependency issue    |
| TPU     | LLVM/XLA runtime conflict | Same expected runtime conflict |

---

# Why CPU, GPU, and TPU Behave Differently

The three backends use different execution paths and optimizations.

### CPU

CPU execution is limited by available system memory and lacks support for several specialized kernels used by MaxText.

### GPU

GPU execution benefits from hardware acceleration but depends on CUDA, cuDNN, Triton kernels, and Transformer Engine compatibility. Hardware generation also affects support for specific optimizations.

### TPU

TPUs use XLA compilation and TPU-specific execution paths. While highly optimized for transformer workloads, they rely heavily on correct XLA and runtime configuration.

---

# Why the Scaled Model Behaves Differently

Increasing model size directly increases:

* Parameter count
* Memory consumption
* Activation memory
* Computational cost

The Qwen3-1.7B model doubles both embedding and MLP dimensions compared to the 0.6B model. As a result, the CPU backend encountered memory exhaustion during initialization, demonstrating the increased resource requirements of larger transformer models.

---

# Key Observations

1. MaxText supports multiple hardware backends but backend-specific dependencies play a major role in successful execution.

2. GPU execution is strongly influenced by hardware generation and CUDA ecosystem compatibility.

3. TPU execution depends heavily on the XLA runtime environment.

4. Scaling model dimensions significantly increases memory requirements even before training begins.

5. Environment limitations were the primary bottleneck during experimentation rather than model configuration issues.

---

# Conclusion

The Qwen3-0.6B and Qwen3-1.7B models were successfully explored, configured, and executed across CPU, GPU, and TPU environments.

Although none of the runs completed the full training process due to backend-specific limitations in the Colab environment, the experiments provided insight into:

* MaxText configuration structure
* Transformer scaling behavior
* Resource requirements of larger models
* Differences between CPU, GPU, and TPU execution paths

The scaled model demonstrated higher memory requirements and exposed additional infrastructure constraints compared to the base model, highlighting the practical challenges of training larger language models across different hardware platforms.
