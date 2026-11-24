# NLLB training recipes

For R4, start with [the training notebook](../notebooks/06_nllb_r4_lora_training.ipynb). It covers both recorded 450-update arms, the loss implementation and a short GPU run on public B01 examples. For the Gemma training loop, see [the 300,000-pair notebook](../notebooks/05_gemma_300k_lora_gpu.ipynb).

## First NLLB training run

The original training and model download scripts are `train_nllb.py` and `model_snapshot.py`. The model translates Polish, Croatian and Russian to English. One shuffled pass over 300,000 official pairs used rank-8 LoRA on the attention query and value projections, an effective batch of 32, a peak learning rate of 0.0001 and 9,375 optimizer updates. Ukrainian was reserved for evaluation. The [evaluation notebook](../notebooks/evaluate_translation.ipynb) can be run independently on CPU.

## Rebuild and check the input

Run the [document-isolated pilot builder](../README.md) with seed 42 and keep the verified output directory. The trainer reads its `train.jsonl.gz`. The original compressed file is 50,990,926 bytes and has SHA-256 `13f1882f113120dfd09d7e5ef7f0c7c834751856266918330d1190741013cf27`. The separate Gemma v2 training file contains different rows.

The trainer checks the compressed digest before using the data. The recorded builder used Python 3.12.14, Unicode 15.0.0 and zlib 1.2.12. If the hash differs, compare the uncompressed JSONL digest (expected `49f606a40d787ba0aca136313d66b47e830a4681c467a1f02627abbcf38057b3`) to distinguish different examples or serialization from different compression. Use the recorded builder environment to recover the pinned archive.

## Run on a single CUDA GPU

Use a GPU runtime with CUDA-enabled PyTorch 2.6 or later. These commands run from the repository root in a terminal. Keep the model cache and training output on a volume that will survive a runtime restart. Set `PILOT_DIR` to the output directory from the pilot builder. The example work directory is relative to the repository root.

```bash
PILOT_DIR="pilot-v1"
RUN_WORK="nllb-work"
mkdir -p "$RUN_WORK"
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print("torch==" + torch.__version__)' > "$RUN_WORK/torch-constraint.txt"
python -m pip install -c "$RUN_WORK/torch-constraint.txt" transformers==4.57.1 peft==0.17.1 tokenizers==0.22.2 sentencepiece==0.2.1
```

The model is [Meta NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M/tree/f8d333a098d19b4fd9a8b18f94170487ad3f821d). Review the [model card](https://huggingface.co/facebook/nllb-200-distilled-600M) and its [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/) before downloading. The first command below downloads about 2.48 GB from the publisher and verifies seven file hashes. The second starts the GPU run.

```bash
python training/model_snapshot.py "$RUN_WORK/base-model"
python training/train_nllb.py --data-dir "$PILOT_DIR" --model-dir "$RUN_WORK/base-model" --run-dir "$RUN_WORK/run" --time-limit-minutes 180
```

The trainer checkpoints at optimizer boundaries. Exit status 75 means it reached the time limit and saved its state. Rerun the same command in the same environment to resume. The final adapter, its configuration and a hash-bearing `TRAINING_COMPLETE.json` are written under `$RUN_WORK/run/training/final_adapter/`. Full comparison with the unadapted model additionally requires generation and scoring on the held-out panels. Metrics from the recorded experiment are in [nllb_everyday.csv](../results/nllb_everyday.csv).
