# Preserving an appointment time in translation

The Polish expression *o wpół do szóstej rano* means 05:30. A translation that says “at 5:30” leaves an English reader to choose between morning and evening. We tested whether a small translation model could preserve that distinction, and whether writing the time in digits before translation helped.

## What was measured

The panel contains 32 synthetic, agent-authored scenarios. Each has two written messages, A and B, which specify different half hours in the same situation. Polish, Croatian, Russian and Ukrainian each contribute eight scenarios. They were authored before model outputs were available and subsequently examined during development. For each language, a source-only rule converts one supported half-hour phrase with a daypart into a 24-hour clock. The replay verifies all 64 replacements against the archived inputs.

Blinded agents saw the English output without the system name, source or reference. They recorded the time or marked it ambiguous, missing, multiple or invalid. Earlier readings could be reused for byte-identical English outputs. A message passes if it gives one unambiguous instant equal to the source time. A scenario passes if both A and B pass. Under this rubric, “07:30” and “7:30 a.m.” identify the same instant. Bare “7:30”, “10:30” and “12:30” leave the daypart unresolved. A qualifier such as “around” can also remove the precision the writer specified.

## Format and adapter on the same messages

| Model and source format | Correct messages, PL/HR/RU | Complete pairs, PL/HR/RU | Correct messages, UK | Complete pairs, UK |
| --- | ---: | ---: | ---: | ---: |
| Base NLLB, colloquial time | 0/48 | 0/24 | 0/16 | 0/8 |
| Base NLLB, digits | 21/48 | 8/24 | 10/16 | 4/8 |
| NLLB + R4, colloquial time | 10/48 | 0/24 | 1/16 | 0/8 |
| NLLB + R4, digits | 46/48 | 22/24 | 16/16 | 8/8 |
| Original Gemma 3 1B-IT, digits | 39/48 | 18/24 | 15/16 | 7/8 |
| OPUS-MT, digits | 5/48 | 1/24 | 3/16 | 0/8 |

On the primary panel, source normalization gained eight full pairs with base NLLB and 22 with R4. Of the 24 normalized pairs, R4 passed five that Gemma missed, Gemma passed one that R4 missed, and both missed one. The separate Ukrainian result includes an explicit Ukrainian parsing rule and zero Ukrainian R4 fine-tuning pairs. These are measurements on the inspected panel. Generalization to other clock constructions and comprehension by human recipients remain open questions.

## What the numeric check caught

We compared the numeric checker with the frozen English readings on all 64 normalized messages per system. The checker returns `match`, `mismatch` or `review`. In this audit, either of the latter two counts as flagged. We scored the original translations before any intervention.

| System | Failed clock readings | Failures flagged | Correct readings flagged | Failures given a match |
| --- | ---: | ---: | ---: | ---: |
| NLLB + R4 | 2 | 2 | 0 | 0 |
| Base NLLB | 33 | 33 | 1 | 0 |
| Gemma 3 1B-IT | 10 | 9 | 0 | 1 |
| OPUS-MT | 56 | 56 | 0 | 0 |

* **Two wrong times from R4.** PL-04-B changed a 04:30 instruction to 4:30 p.m. HR-01-A moved filming nocturnal animals from 00:30 to 1:30 p.m., thirteen hours late. Gemma preserved 00:30 in the latter case.
* **A numeric match that changes the instruction.** In PL-07-B, Gemma wrote “around 9:30 PM” for a subtitle test scheduled at exactly 21:30. The checker matched the digits while the blinded reader marked the time ambiguous.
* **Correct time, different task.** In PL-01-A, R4 rendered the unloading hose as “the snake”. In PL-02-A, Gemma released squirrels for training and then asked for the number of birds released. Both messages passed their respective clock checks.

## Why the metric ranking changes

On the 48 primary messages, switching R4 from colloquial to normalized input moved mean COMET from 0.79107 to 0.78974 while complete clock pairs rose from 0 to 22. COMET used the same original source and English reference for both outputs. Raw corpus BLEU moved from 22.30 to 21.63. All 64 references use 24-hour digits. When we rewrote a single explicit English AM/PM clock in each eligible prediction to the same notation, BLEU moved from 34.11 to 43.78. Standardizing notation preserves the hour the model actually gave, including a wrong hour.

The same diagnostic changes the Ukrainian ranking. On the 16 Ukrainian messages, raw BLEU was 31.65 for Gemma and 24.82 for R4. After applying the identical formatting rule, the scores were 46.91 and 48.60. The case readings and literal translations show which errors could change a reader's decision. [Corpus scores by language, model and notation](../results/norm64_text_metrics.csv)

## Reproduce the analysis

From the repository root, Python 3.9 or later can replay normalization, score the saved readings, and run the numeric checker. The JSON result includes per-language counts and the case IDs of disagreements between the checker and the blinded readings.

```sh
python3 pipeline/score_norm64.py --output /tmp/norm64.json
python3 -m unittest discover -s tests -v
```

To reproduce the metric CSV, run the following in a virtual environment. The script computes corpus BLEU and chrF++ from translations and references and averages the archived per-message COMET scores for the two R4 runs.

```sh
python3 -m pip install sacrebleu==2.5.1
python3 pipeline/norm64_text_metrics.py > /tmp/norm64_text_metrics.csv
diff -u results/norm64_text_metrics.csv /tmp/norm64_text_metrics.csv
```

The case file includes both source forms, a reference, six saved outputs per message, the blinded readings and the expected time. The run manifest identifies the models, adapter, decoding settings and hardware. Regenerating R4 translations requires the matching adapter weights separately. Gemma ran on a Colab CPU and the MT models ran on Colab GPUs, so these runs do not establish a latency ranking or phone performance. [Case file](../results/norm64_cases.jsonl) · [Run manifest](../results/norm64_run.json) · [Source normalizer](../pipeline/clock_normalize.py) · [Numeric checker](../pipeline/clock_check.py)
