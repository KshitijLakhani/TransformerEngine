# Transformer Engine Examples #

This folder contains simple examples introducing Transformer Engine and FP8 training usage.

**Examples Outline**
* MNIST training: Training MNIST dataset is a good start point to learn how use Transformer Engine and enable FP8 training
* Encoder training: The encoder examples introduce more about how to scale up training on multiple GPUs with Transformer Engine
* Fused Attention: The attention examples demonstrate how to use TE's cuDNN fused attention — self-attention with GQA and bias, cross-attention with THD packing and sliding window, and context parallelism across multiple GPUs