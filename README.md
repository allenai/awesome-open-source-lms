# Awesome Open (Source) Language Models

Friends of OLMo and their links. Built for the [2024 NeurIPS tutorial on opening the language modeling pipeline](https://neurips.cc/virtual/2024/tutorial/99526) by Ai2 (slides [here](https://docs.google.com/presentation/d/179dpzWSQ9G7EAUlvaJdeE0av9PLuk9Rl33nfhHSJ4xI/edit?usp=sharing)).
> Language models (LMs) have become a critical technology for tackling a wide range of natural language processing tasks, making them ubiquitous in both AI
research and commercial products.
> As their commercial importance has surged, the most powerful models have become more secretive, gated behind proprietary interfaces, with important details of their training data, architectures, and development undisclosed. 
> Given the importance of these details in scientifically studying these models, including their biases and potential risks, we believe it is essential for the research community to have access to powerful, truly open LMs. 
> In this tutorial, we provide a detailed walkthrough of the language model development pipeline, including pretraining data, model architecture and training, adaptation (e.g., instruction tuning, RLHF). 
> For each of these development stages, we provide examples using open software and data, and discuss tips, tricks, pitfalls, and otherwise often inaccessible details about the full language model pipeline that we've uncovered in our own efforts to develop open models. 
> We have opted not to have the optional panel given the extensive technical details and examples we need to include to cover this topic exhaustively.

This focuses on language models with **more than just model weights being open** -- looking for training code, data, and more! 
The best is fully **open-source** language models with the entire pipeline, but individual pieces are super valuable too.

🚧 Missed something? Give us a PR to add! 🚧

---

## OLMo 2 (Nov. 2024)

- [Collection](https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc)  
- [7B base model](https://huggingface.co/allenai/OLMo-2-1124-7B)  
- [13B base model](https://huggingface.co/allenai/OLMo-2-1124-13B)  
- [7B instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)  
- [13B instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct)  
- [Annealing dataset](https://huggingface.co/datasets/allenai/dolmino-mix-1124)  
- [Training Code (1st gen.)](https://github.com/allenai/OLMo)  
- [Training Code (2nd gen.)](https://github.com/allenai/OLMo-core)  
- [Post-train Code](https://github.com/allenai/open-instruct)  
- [Eval Code](https://github.com/allenai/olmes)  
- [Data Processing Toolkit](https://github.com/allenai/dolma)  
- [Demo](https://playground.allenai.org/)

## AMD OLMo (Oct. 2024)

- [1B model](https://huggingface.co/amd/AMD-OLMo)
  - [SFT](https://huggingface.co/amd/AMD-OLMo-1B-SFT)
  - [SFT-DPO](https://huggingface.co/amd/AMD-OLMo-1B-SFT-DPO)
- [Blog post](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html)

## HuggingFace SmolLM (v2 Oct. 2024)

- [SmolLM 2 collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)  
- SmolLM 2 pretraining data: TBD  
- [SmolLM instruction mix](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)  
- [SmolLM collection](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)  
- [SmolLM pretraining data](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)  
- [Synthetic pretrain corpus](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)  
- [Fineweb pretrain corpus](https://huggingface.co/datasets/HuggingFaceFW/fineweb)  
  - [Edu Subset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)  
  - [Fineweb 2 (multilingual)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- [SmolLM repo](https://github.com/huggingface/smollm)
  - [Training Code](https://github.com/huggingface/nanotron)
  - [Eval Code](https://github.com/huggingface/lighteval)  
  - [Data Processing Code](https://github.com/huggingface/datatrove)
- Blogposts:  
  - [Fineweb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)  
  - [Fineweb v2](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks)  
  - [SmolLM](https://huggingface.co/blog/smollm)

## DataComp (Jun. 2024)

- [1B Model](https://huggingface.co/TRI-ML/DCLM-1B)  
  - [Instruct](https://huggingface.co/TRI-ML/DCLM-1B-IT)  
- [7B Model](https://huggingface.co/apple/DCLM-7B)  
  - [Instruct](https://huggingface.co/mlfoundations/dclm-7b-it)  
  - [8K Extension](https://huggingface.co/apple/DCLM-7B-8k)  
- [Pretrain Dataset](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)  
- [Training + Eval Code](https://github.com/mlfoundations/dclm)  
- [Paper](https://arxiv.org/abs/2406.11794)
- [Evalchemy (Post-training Eval Code)](https://github.com/mlfoundations/evalchemy)

## Databricks / formerly Mosaic ML

- [Streaming Datasets](https://github.com/mosaicml/streaming)  
- [Eval Gauntlet](https://github.com/mosaicml/llm-foundry/blob/main/scripts/eval/local_data/EVAL_GAUNTLET.md)  
- [Pretraining Code](https://github.com/mosaicml/composer)  
  - [Megablocks (MoE Training)](https://github.com/databricks/megablocks)
- [DBRX Collection](https://huggingface.co/collections/databricks/dbrx-6601c0852a0cdd3c59f71962)

## LLM 360

- [Analysis360](https://github.com/LLM360/Analysis360)  
- K2-65B:  
  - [Collection](https://huggingface.co/collections/LLM360/k2-6622ae6911e3eb6219690039)  
  - [Data](https://huggingface.co/datasets/LLM360/K2Datasets)  
  - [Data Preparation Code](https://github.com/LLM360/k2-data-prep)  
  - [Training Code](https://github.com/LLM360/k2-train)  
- CrystalCoder-7B:  
  - [Data](https://huggingface.co/datasets/LLM360/CrystalCoderDatasets)  
  - [Data Preparation Code](https://github.com/LLM360/crystalcoder-data-prep)  
  - [Training Code](https://github.com/LLM360/crystalcoder-train)  
- Amber-7B:  
  - [Collection](https://huggingface.co/collections/LLM360/amber-65e7333ff73c7bbb014f2f2f)  
  - [Data](https://huggingface.co/datasets/LLM360/AmberDatasets)  
  - [Data Preparation Code](https://github.com/LLM360/amber-data-prep)  
  - [Training Code](https://github.com/LLM360/amber-train)  
- [Paper](https://arxiv.org/abs/2312.06550)

## EleutherAI

- Pythia
  - [Models Collection](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1)
  - [Central Repo](https://github.com/EleutherAI/pythia)
  - [Paper](https://arxiv.org/abs/2304.01373)

- GPT-NeoX-20B
  - [Model](https://huggingface.co/EleutherAI/gpt-neox-20b)
  - [Paper](https://arxiv.org/abs/2204.06745)

- Llema-7B
  - [Central Repo](https://github.com/EleutherAI/math-lm)
  - [Paper](https://arxiv.org/abs/2310.10631)
  - [Dataset](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
  - [Blogpost](https://blog.eleuther.ai/llemma/)



- [Training Code](https://github.com/EleutherAI/gpt-neox)  
- [Eval Code](https://github.com/EleutherAI/lm-evaluation-harness)
  - [Paper](https://arxiv.org/abs/2405.14782)
  
- [The Pile Dataset](https://huggingface.co/datasets/EleutherAI/pile)
  - [Paper](https://arxiv.org/abs/2101.00027)


## Cerebras

- Cerebras-GPT
  - [Models Collection](https://huggingface.co/collections/cerebras/cerebras-gpt-66c623297a2370b8e670e0a1)
  - [Paper](https://arxiv.org/abs/2304.03208)
  - [Blogpost](https://cerebras.ai/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/)
  - [Training Code](https://github.com/Cerebras/modelzoo)


## RWKV

- [RWKV-5 Models Collection](https://huggingface.co/collections/RWKV/rwkv-v5-6633e58adbe81b1466f9676b)
- [RWKV-6 Models Collection](https://huggingface.co/collections/RWKV/rwkv-v6-669cb221fe9496b3c693c8e9)
- [Training Code](https://github.com/BlinkDL/RWKV-LM)
- Papers:
  - [RWKV-4](https://arxiv.org/abs/2305.13048)
  - [RWKV-5 and RWKV-6](https://arxiv.org/abs/2404.05892)


## M.A.P.

- [Models Collection](https://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04)  
- [Datasets](https://huggingface.co/collections/m-a-p/neo-datasets-66395dc55cbebc0a7767bbd5)  
- [Paper](https://arxiv.org/abs/2405.19327)  
- [Code](https://github.com/multimodal-art-projection/MAP-NEO)

## Zyphra

- Zamba 2 Models:  
  - [7B](https://huggingface.co/Zyphra/Zamba2-7B)  
    - [7B Instruct](https://huggingface.co/Zyphra/Zamba2-7B-Instruct)
    - [Blogpost](https://www.zyphra.com/post/zamba2-7b)
  - [2.7B](https://huggingface.co/Zyphra/Zamba2-2.7B)  
    - [Instruct](https://huggingface.co/Zyphra/Zamba2-2.7B-Instruct)
    - [Blogpost](https://www.zyphra.com/post/zamba2-small)
  - [1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B)  
    - [Instruct](https://huggingface.co/Zyphra/Zamba2-1.2B-Instruct)
    - [Blogpost](https://www.zyphra.com/post/zamba2-mini)
  - [Paper](https://arxiv.org/abs/2411.15242)
- [Zyda 2 Dataset](https://huggingface.co/datasets/Zyphra/Zyda-2)  
  - [Blogpost](https://www.zyphra.com/post/building-zyda-2#zyda2_7)
  - [Paper](https://arxiv.org/abs/2411.06068)

## Together.AI

- [RedPajama v1 Dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
- [RedPajama v2 Dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)
- [Data Code](https://github.com/togethercomputer/RedPajama-Data)

## Inference.net

- Schematron Models (HTML-to-JSON Extraction):
  - [Schematron-3B](https://huggingface.co/inference-net/Schematron-3B)
    - Specialized HTML-to-JSON extraction model with 128K context window
    - Base model: Meta-Llama-3.2-3B-Instruct
    - 3B parameters, BF16 precision
    - [Model Card](https://huggingface.co/inference-net/Schematron-3B) | [Documentation](https://docs.inference.net/use-cases/json-extraction)
  - [Schematron-8B](https://huggingface.co/inference-net/Schematron-8B)
    - High-accuracy HTML-to-JSON extraction with marginal quality lift
    - Base model: Meta-Llama-3.1-8B-Instruct
    - 8B parameters, BF16 precision
    - [Model Card](https://huggingface.co/inference-net/Schematron-8B) | [Documentation](https://docs.inference.net/use-cases/json-extraction)
  - [Serverless API](https://docs.inference.net/use-cases/json-extraction)
  - [Blog Post: Structured HTML Extraction](https://inference.net/blog/Schematron)

- ClipTagger-12b (Video Frame Captioning):
  - [ClipTagger-12b](https://huggingface.co/inference-net/ClipTagger-12b)
    - 12B parameter vision-language model for video frame captioning
    - Base architecture: Gemma-12B with FP8 quantization
    - Specialized for structured JSON output from video frames
    - Cost-efficient: 15x cheaper than GPT-4.1, 17x cheaper than Claude 4 Sonnet
    - [Model Card](https://huggingface.co/inference-net/ClipTagger-12b) | [Documentation](https://docs.inference.net/use-cases/video-understanding)
  - [Blog Post: Video Understanding at Scale](https://inference.net/blog/cliptagger-12b)

## NVIDIA

- Pretraining Code
  - [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
  - [NeMo](https://github.com/NVIDIA/NeMo)  
- [Data Processing](https://github.com/NVIDIA/NeMo-Curator)  
- [Post-training Code](https://github.com/NVIDIA/NeMo-Aligner)

## PyTorch / Meta

- [Torch Titan Pretraining Code](https://github.com/pytorch/torchtitan)  
- [Meta Lingua Pretraining Code](https://github.com/facebookresearch/lingua)

## Open-sci, OpenEuroLLM, LAION AI (Sept. 2025)
- [Paper](https://www.arxiv.org/pdf/2509.09009)
- [Blog](https://laion.ai/blog/open-sci-ref-001/)
- [HF collections](https://huggingface.co/collections/open-sci/open-sci-ref-001-685905e598be658fbcebff4f)
