# Slavic–English multilingual neural machine translation

Technical specification and research proposal · Public edition · 23 November 2026

## 1. Executive summary and historical precedent

A Polish school message asks whether a daughter can answer a question after class. In our test, an English translation addressed that request to the teacher instead. The sentence was fluent. A teacher reading it could think the question was theirs to answer. This project studies English translations of everyday messages from people who have moved abroad.

We are testing whether a compact translator can learn from everyday examples checked by bilingual speakers to preserve who should do what, when and under what conditions. We chose Polish, Russian and Croatian because each represents a different branch of the Slavic family. All three use case endings to signal who did what to whom. Training one model to express those distinctions in English may also help with related languages that have fewer translated documents. Ukrainian gives us a first test of that idea because we held it out of adapter training.

The first NLLB experiment and a later one-billion-parameter Gemma experiment each used 300,000 official translation pairs. Both improved performance on institutional language. On 72 everyday messages, the full Gemma run raised mean COMET in Croatian and lowered it in Polish, Russian and Ukrainian. A smaller Gemma experiment tested reviewed everyday training examples at a matched number of steps. Its results vary by language. Useful translation in this setting depends on what an English-speaking recipient actually understands, which remains to be measured.

### Problem statement

**When people who have moved abroad send everyday messages in English, in which situations does a translation help the recipient understand their intent more accurately than their usual way of communicating?**

For a future participant study, each writer will choose a situation from their own life and record who is expected to act, what conditions apply and what they want to leave open. Different readers will see either the version prepared in the writer’s usual way or the model translation. Each reader will explain what they understood and what they would do or ask next. The writer will judge whether that interpretation preserves their intent. If the writer meant to leave a decision open, a clarifying question can be the right outcome.

### Capstone pathway and project category

**Translation and meaning preservation.** This project adapted the summarization pathway of the Google DeepMind AI Research Foundations capstone to multilingual translation. The output is an English rendering of the source message. Conversation supplies the setting. Classification enters the evaluation when reviewers judge whether a condition was preserved, changed or left open. The training and evaluation work described here was conducted for this independent project. [Course repository](https://github.com/google-deepmind/ai-foundations)

### Why start with parliamentary records

The English–French Canadian Hansards are an example of professionally translated parliamentary proceedings used in translation research. They motivated our use of UN and EU material, where aligned text is available across languages. Official translations supply references for institutional language. Alignment quality and everyday meaning still require separate checks. [Canadian Hansards corpus](https://huggingface.co/datasets/usc-isi/hansards)

<!-- page -->

## 2. Strategic Anchor Framework

We call Polish, Russian and Croatian the anchor languages because each contributes enough parallel text to adapt a shared model into English. Balanced sampling prevents the largest archive from dominating the training run. NLLB generation selects the target tag `eng_Latn`. The decoder-only Gemma experiments request English in the translation instruction.

| Anchor and NLLB identifier | Training resources and transfer hypothesis |
|---|---|
| **Polish · West Slavic** · Latin with diacritics · `pol_Latn` | EU statutes and parliamentary proceedings. Polish syntax and vocabulary may help Czech, Slovak, and Upper and Lower Sorbian. The earlier catalogue also named [PolEval](https://poleval.pl/) and [BEIR-PL](https://arxiv.org/abs/2305.19840). These are task/benchmark resources rather than verified sources of Polish–English training pairs. |
| **Russian · East Slavic** · Cyrillic · `rus_Cyrl` | UN parallel text provides the formal Russian anchor. Vocabulary, case patterns and script may help Ukrainian (`ukr_Cyrl`) and Belarusian. OpenSubtitles and ParaCrawl are candidate supplements. Only Ukrainian has been tested for transfer. |
| **Croatian · South Slavic** · Gaj’s Latin alphabet · `hrv_Latn` | EU DGT was the Croatian source actually used. Verbal aspect and enclitic patterns are of interest for Serbian, Bosnian, Slovenian and Bulgarian. The project first proposes matched tests for Serbian and Slovenian. |

The first pilot trains three directions, PL→EN, RU→EN and HR→EN. The base NLLB already supports Ukrainian. Here *zero-shot* means no Ukrainian examples during our adapter training. The operation of the base model and the genealogy of each language by themselves establish no quality threshold for relatives.

## 3. Shared representations and false friends

One multilingual encoder and a shared English decoder can reuse representations across inputs. Polish *woda*, Russian *вода* and Croatian *voda* all mean “water”. The interlingua hypothesis is that multilingual pretraining helps the encoder represent related meanings in ways the same decoder can use, reducing the need for separate pairwise models. Nearly identical embeddings would require direct measurement.

Gender agreement, case marking and perfective/imperfective aspect offer other possible cues. Many Slavic languages have six or seven cases, with important differences across branches. Shared subwords may capture parts of roots and affixes. Similar segmentation across Latin and Cyrillic, or preservation of morpheme boundaries, must be checked on actual text.

### Context must distinguish similar forms

| Source words | Different meanings the English translation must preserve |
|---|---|
| Ukrainian **вродливий** / Russian **уродливый** | Attractive or handsome / ugly or deformed |
| Ukrainian **неділя** / Russian **неделя** | Sunday / a week |
| Ukrainian **пильний** / Russian **пыльный** | Vigilant or alert / dusty |
| Polish **sklep** / Russian or Ukrainian **склеп** | Shop / burial vault or crypt in the tested sense |
| Polish **zapomnieć** / Russian **запомнить** | To forget / to remember |

An NLLB encoder can use the sentence and its source-language tag when contextualizing a word. A decoder-only model sees the language named in the prompt. Section 7 proposes a contrastive diagnostic to distinguish correct and incorrect English readings in matched contexts.

<!-- page -->

## 4. Primary Parallel Corpora & Dataset Catalog

The original pilot draws on four versioned archives. Counts below are specific to the release and processing stage shown. An aligned row can contain an empty segment or corrupted text. The right column reports the pairs that actually entered each of the full training runs.

| Source, pair and register | Raw records and training contribution |
|---|---|
| [UN Parallel Corpus v1.0](https://www.un.org/dgacm/en/content/uncorpus) · RU–EN · diplomatic and institutional texts | The OPUS release has 25,173,398 aligned rows, 87.3% of raw rows across the four archives. We sampled 100,000 Russian training pairs. |
| [Europarl v10](https://www.statmt.org/europarl/v10/) · PL–EN · parliamentary debates | Repairing the malformed TSV rows yielded 630,768 nonempty PL–EN pairs. We used 28,000 for training. Croatian Europarl was a candidate in the plan. The Croatian pilot used DGT. |
| [JRC-Acquis v3.0](https://joint-research-centre.ec.europa.eu/language-technology-resources/jrc-acquis_en) · PL–EN · EU statutes | The PL–EN archive has 1,610,962 aligned rows. We used 72,000 for training. The project did not establish the originally proposed Croatian coverage for this archive. |
| [DGT Translation Memory](https://joint-research-centre.ec.europa.eu/language-technology-resources/dgt-translation-memory_en) · OPUS v2021 · HR–EN · EU institutional text | The Croatian source has 1,416,461 aligned rows in 8,147 document groups. We checked the parallel text against the alignment index and used 100,000 pairs. |

### Other resources in the design

The original plan considered broader resources. Their usefulness depends on the specific language pair, release, quality and reuse conditions. The following collections contributed zero training pairs to the full runs.

| Additional resource | Intended purpose and qualification |
|---|---|
| [OPUS](https://opus.nlpl.eu/) and [ParaCrawl](https://paracrawl.eu/) | OPUS is a catalogue. ParaCrawl is a candidate source of web bitext. Audit each language release for available aligned pairs, duplicates, licensing and noise before sampling. |
| [OPUS OpenSubtitles](https://aclanthology.org/L16-1147/) | A potential source of dialogue. Verify language-pair coverage, alignment and rights. Scripted subtitle exchanges can differ from a real housing or workplace request. |
| [CLASSLA-web 2.0](https://clarinsi.github.io/classla-web/) | A collection of comparable South Slavic web corpora. Its reported 17 billion words span seven languages and include genre and linguistic annotation. The word count measures monolingual coverage, not aligned English translations. |
| [Canadian Hansards](https://huggingface.co/datasets/usc-isi/hansards) | The English–French parliamentary corpus informed the choice of professionally translated proceedings. It supplies no Slavic–English training pairs. |

### Sampling, filtering and leakage checks

The pilot assigns whole documents before sampling and shares an EU identifier across JRC and DGT. The original splits contain 100,000 train, 1,000 validation and 1,000 test pairs per trained language. No document ID occurs in two splits. Exact normalized text, pattern and near-duplicate audits cover specified overlap risks. Release v2 keeps the held-out official pools fixed and replaces 575 corrupted DGT training rows and one concatenated UNPC staffing table with unused alignments from existing training documents. Of 300,000 training rows, 299,424 remain unchanged. Semantic alignment error, unseen paraphrases and exposure during base-model pretraining remain open.

<!-- page -->

## 5. Proposed System Architecture & Tokenisation Design

### NLLB encoder–decoder training

The pilot uses the pretrained NLLB-200-distilled-600M sequence-to-sequence model with its native multilingual tokenizer. The original weights remain frozen. A LoRA adapter learns low-rank adjustments to the attention query and value projections (rank 8, alpha 16, dropout 0.05). All three languages share the decoder and English target. One shuffled epoch over 300,000 pairs used seed 42, effective batch size 32 (four pairs × eight accumulation steps), learning rate 0.0001 and 9,375 optimizer steps. The small adapter must be loaded with the full base model for inference.

NLLB starts from a previously distilled encoder–decoder. Our update uses the ordinary sequence-to-sequence objective. Teacher–student distillation, a separate cross-lingual alignment loss and label smoothing of 0.1 were options in the original design. [NLLB base model card](https://huggingface.co/facebook/nllb-200-distilled-600M)

### Architectural alternatives

| Proposed option | What would justify testing it |
|---|---|
| **A. Distilled encoder–decoder.** Distil a large multilingual teacher such as NLLB-200 or mT5 into a specialised 600M–1.2B Slavic–English student. | Test output quality, source-to-target alignment, decoding latency and memory. The 600M NLLB adapter run provides a working comparison. No separate mT5 student has been trained. |
| **B. Decoder-only SLM.** The design considered Gemma-2-2B or Qwen2.5-3B with multilingual LoRA. | We later fine-tuned Gemma 3 1B-IT in a matched small-data trial and a full official-text run. The separate Gemma 4 E4B-it comparison used a larger instruction model for inference only. |

### Tokenization and language control

The proposed redesign is a shared **48,000–64,000-piece SentencePiece BPE vocabulary**, with balanced representation of Latin and Cyrillic text. Some English-oriented vocabularies may split a single inflected Slavic word into several tokens. The original estimates of four to six pieces per affected word and 35–45% shorter sequences were hypotheses. Measure token fertility by language and script on the same text, then measure translation quality and latency. A new vocabulary also requires compatible embeddings and retraining.

The original explicit task prefixes were `__src_pol__ __tgt_eng__`, `__src_rus__ __tgt_eng__` and `__src_hrv__ __tgt_eng__`. In the working pilot the tokenizer uses the native source tags in section 2 and generation selects `eng_Latn` as the English target. The proposed task strings and custom vocabulary did not enter this run.

### Inference and portability

The paired NLLB benchmark used deterministic decoding with one beam, no sampling, a 512-token source limit and up to 256 new tokens. Mobile packaging requires a runtime that supports encoder–decoder generation, language tokens and the adapter or verified merged weights. The Gemma 3 full run used its native chat format, bfloat16 and greedy evaluation. The earlier quantized Gemma 4 E4B-it comparison was an inference experiment. Phone latency, memory and battery use await direct testing.

<!-- page -->

### How NLLB reads the same request in two scripts

Imagine sending a roommate the instruction “Leave the keys with the neighbor.” Each version below uses four words, a familiar singular imperative and a final period. The table shows the pieces returned by the pilot tokenizer in their original order. The visible mark `▁` signals a word boundary. Each space between items in the middle column separates two tokens.

| Message (four words) | BPE pieces for the words | Pieces per word |
|---|---|---|
| **Russian** · Оставь ключи у соседа. | `▁О став ь ▁клю чи ▁у ▁со седа` | 8 / 4 = **2.00** |
| **Polish** · Zostaw klucze u sąsiada. | `▁Zosta w ▁klu cze ▁u ▁są si ada` | 8 / 4 = **2.00** |
| **Croatian** · Ostavi ključeve kod susjeda. | `▁Osta vi ▁klju če ve ▁kod ▁sus jeda` | 8 / 4 = **2.00** |
| **Ukrainian** · Залиш ключі в сусіда. | `▁За лиш ▁клю чі ▁в ▁сусі да` | 7 / 4 = **1.75** |

**What the counts include.** We tokenized each complete sentence with its source tag. To measure how the words split, we divided the number of word pieces by the four whitespace-delimited words. For these inputs, the source tag, the final period and the end-of-sequence token `</s>` each occupy one additional position. The full encoder input therefore has 11 tokens in Russian, Polish and Croatian, and 10 in Ukrainian. Decoding the IDs reproduced each original sentence.

**How the model handles Cyrillic and Latin.** NLLB reads each language in its usual script. The input starts with `rus_Cyrl`, `pol_Latn`, `hrv_Latn` or `ukr_Cyrl`. We select `eng_Latn` for the English output. All four languages use one vocabulary. Russian *ключи* and Ukrainian *ключі* share the piece `▁клю` and differ in the ending. Polish *klucze* and Croatian *ključeve* start with `▁klu` and `▁klju`, separate entries in that same vocabulary.

The model learns from context and multilingual training how different pieces can lead to the English word *keys*. The tokenizer handles written forms without a transliteration step. The base and adapted model used exactly the same pieces. The four ratios describe this instruction only. Testing the phase 2 target of 1.4 pieces per word requires a larger, matched sample for each language.

We used the fast tokenizer from the pilot checkpoint (`f8d333a`) with Transformers 4.57.1, its native language-tag settings and the original tokenizer files. [Pinned NLLB checkpoint](https://huggingface.co/facebook/nllb-200-distilled-600M/tree/f8d333a098d19b4fd9a8b18f94170487ad3f821d) · [SentencePiece word-boundary convention](https://github.com/google/sentencepiece#whitespace-is-treated-as-a-basic-symbol)

<!-- page -->

## 6. Phased Implementation Roadmap

The week ranges are the original planning estimates. The status below reflects the experiments completed by 23 November. Sections 5 and 7 distinguish the training runs, evidence and remaining acceptance conditions.

### Phase 1 · Weeks 1–2 · Data harvesting and rigorous filtering

**Partial.** The plan called for Europarl (PL and HR), UNPC (RU) and JRC-Acquis, using Bicleaner and LASER embedding-distance checks to remove misalignment and machine-translated web noise. **The gate requires under 0.1% semantic alignment error on validation data.** Four versioned archives were built into a document-isolated pilot. Source rows and document provenance were verified for all 306,000 pairs. Croatian came from DGT. Bicleaner, LASER and the semantic error threshold have no completed validation here. Audit a bilingual validation sample and report the sampling unit and uncertainty before claiming the gate.

### Phase 2 · Weeks 3–4 · Tokenizer and lexicon optimization

**Deferred.** The proposed 48k SentencePiece BPE tokenizer used temperature sampling with exponent 0.3 to balance languages. Evaluate subword fertility for Polish diacritics and Croatian enclitics and inspect all three languages. **The target is at most 1.4 subword tokens per Slavic word.** The pilot retained the native NLLB tokenizer. Section 5 illustrates the count on one matched instruction. The corpus-level fertility comparison remains open.

### Phase 3 · Weeks 5–7 · Joint anchor-model training

**NLLB and Gemma pilot runs complete. Retention gate open.** The NLLB adapter completed 9,375 steps. Gemma 3 1B-IT completed 9,376 steps on its cleaned official training release. The design proposed label smoothing and auxiliary embedding alignment as additional objectives. The completed runs did not use those additions. The original acceptance condition requires stable loss and retention across languages. A single epoch on three anchors shows progress on its training distribution. Broader retention requires a prespecified evaluation set.

### Phase 4 · Weeks 8–9 · Zero-shot evaluation on relatives

**Started.** The planned languages are Ukrainian, Czech, Slovak, Serbian and Slovenian. NLLB has paired results on 1,012 Ukrainian FLORES-200 devtest sentences. The full Gemma evaluation used a frozen subset of 32. Ukrainian had zero pairs in adapter training. The original gate, 85% of a supervised anchor score, still needs a matched metric, language direction, reference and domain. Additional relative-language and supervised comparisons remain open.

### Phase 5 · Weeks 10–12 · Quantization and edge deployment

**Planned.** Evaluate 8-bit and 4-bit weights and an appropriate export/runtime, including the originally proposed GGUF, AWQ and ONNX routes where compatible. Test offline laptop and phone inference. **The proposed limits are 25 ms per generated token and 3.5 GB of accelerator memory.** There is no verified handset run. Name the device and runtime, report peak total working memory on unified-memory phones, time to first translation, model download size and battery cost. Recheck adequacy on the same messages before proposing deployment.

<!-- page -->

## 7. Evaluation Suite & Quality Metrics

### What each measure answers

**BLEU and chrF++.** Corpus BLEU measures word n-gram overlap with an English reference. chrF++ combines character and word n-grams and can recognise partial word matches. In this project both are measured on English outputs. Neither directly shows that a reader understood the intended condition. We report SacreBLEU scores on the same frozen inputs with matching decoding. Scores from different domains answer different questions.

**COMET and xCOMET-lite.** COMET scores a translation using its source and English reference. This helps us assess faithful paraphrases that share few words with the reference. We used Unbabel/wmt22-comet-da. xCOMET also predicts the location and severity of suspected errors. We tested the open distilled myyycroft/XCOMET-lite on the same everyday messages and compared its spans with errors found in blind, model-assisted review. [COMET](https://aclanthology.org/2020.emnlp-main.213/) · [xCOMET](https://aclanthology.org/2024.tacl-1.54/) · [xCOMET-lite](https://aclanthology.org/2024.emnlp-main.1223/)

**Human adequacy and fluency.** A bilingual reviewer checks that the original sounds natural and that the English retains the actor, action, timing, degree of certainty and exceptions. A recipient-level test hides the translator’s identity and asks what a reader would do. The author assesses that interpretation. Approval of proposed training pairs and blind evaluation on held-out scenarios are separate tasks.

**Diagnostic false-friend set.** The design calls for 500 contrastive sentences using lexical traps such as those in section 3. Balance contexts across languages and meanings, obtain bilingual keys, and test whether the English equivalent is right for each context. A 500-sentence diagnostic has not been established. The 72 everyday messages and 16 stress cases below are distinct completed tests.

### Measured change after the first anchor training

| Source | Benchmark BLEU · base → adapter | Benchmark chrF++ · base → adapter | Major everyday errors · base → adapter |
|---|---|---|---|
| Croatian | 37.01 → 55.27 | 57.01 → 71.47 | 14 → 11 of 18 |
| Polish | 44.05 → 48.05 | 63.74 → 66.82 | 17 → 13 of 18 |
| Russian | 36.84 → 40.71 | 61.72 → 63.56 | 11 → 11 of 18 |
| Ukrainian | 36.80 → 36.51 | 60.27 → 60.64 | 17 → 9 of 18 |

The paired benchmark contains 1,000 official-text segments per trained language and 1,012 general-domain FLORES-200 devtest sentences for Ukrainian. Eight Croatian test records contained corrupted characters. Excluding them gave NLLB BLEU 37.21 → 55.56 on 992 records. DGT terminology recurs across institutional documents. Isolating the effects of domain and shared multilingual training would require matched domains and monolingual controls.

Ukrainian contributed zero pairs to this fine-tuning. The NLLB everyday error count improved on a small authored set. On FLORES, BLEU declined slightly and the paired 95% interval for the small chrF++ gain included zero. Transfer beyond these examples remains uncertain. [NLLB experiment and settings](https://github.com/Silveroboros-dev/slavic-english-slm/tree/main/training)

<!-- page -->

### Everyday decisions and the first stress test

We compared the base model, its adapter, Gemma 4 E4B-it and online Google Translate on the same 72 authored messages, 18 in each language. A separate 16-case stress test used four deliberately difficult cases per language, covering conditions and order of actions, numerical boundaries, what a document establishes, and a word’s meaning in context. Answers and keys were fixed before evaluation. Both sets used preliminary model-assisted review. Neither measures how common the errors are in the community.

| Translator | Major meaning errors in everyday set (of 72) | Full meaning preserved in stress set (of 16) | All specified questions answered correctly (of 16) |
|---|---|---|---|
| Base NLLB | 59 | 4 | 9 |
| NLLB with LoRA | 44 | 4 | 8 |
| Gemma 4 E4B-it | 14 | 14 | 16 |
| Google Translate | 8 | 14 | 15 |

Of the base model’s 59 major everyday errors, the adapter resolved 17 and introduced two, leaving 44.

A Polish repair case changed who had to confirm a visit. The source assigned confirmation to the tenant and the adapter assigned it to the dispatcher. In a Russian parcel case, Gemma 4 E4B-it preserved the answers to the narrow questions while broadening the required passport to a generic ID. A correct answer to one question can coexist with an important error elsewhere in the message.

### What the metric missed

On these 72 messages, the two scorers produced the following mean segment scores. Higher values indicate better expected translation quality for each metric.

| Translator | Mean COMET | Mean xCOMET-lite |
|---|---|---|
| Base NLLB | 0.7260 | 0.8110 |
| NLLB with LoRA | 0.7700 | 0.8696 |
| Gemma 4 E4B-it | 0.8510 | 0.9367 |
| Google Translate | 0.8608 | 0.9443 |

xCOMET-lite returned no error spans across all 288 translations, including the three newly introduced Gemma errors examined (hr-16, ru-11 and uk-18). The released check produced no span even for deliberately altered controls. Its sentence scores varied with the translations, but an empty span did not establish accuracy. For the next stress test, retain the blind adequacy judgments and the reader’s answer alongside metric scores.

<!-- page -->

### A payslip or a timesheet? One term changes the document

In a Croatian workplace case from the original authored test, a worker reported a discrepancy between the four Sunday hours on her payslip and the six she worked. Two recorded translations preserved both numbers but named different records. The scores below show why we compare metrics with the decision a recipient might make.

**Source expression.** The Croatian phrase *na platnoj listi* refers to a payslip in this context. The project English reference used *payslip*.

**Google Translate.** The critical term in its English output was *pay slip*, spelled as two words.

**Gemma 4 E4B-it.** The critical term was *timesheet*, which would direct a reader to an attendance record.



### Which record would the recipient check?

An English-speaking recipient of the Google translation might check the worker’s payslip for the Sunday hours. Gemma’s wording would point the recipient to the timesheet used to record attendance. Both translations preserve four hours recorded and six worked. The worker identifies a discrepancy on the payslip. Its cause and any effect on her wages are unknown.

| Translator | Sentence BLEU | COMET | xCOMET-lite |
|---|---|---|---|
| Google Translate | 80.53 | 0.9475 | 0.9613 |
| Gemma 4 E4B-it | 86.12 | 0.9149 | 0.9263 |

**Why BLEU preferred “timesheet”.** The reference spells *payslip* as one word. Google’s equivalent *pay slip* takes two tokens. Both candidates matched the same 14 reference unigrams, but Google had 17 tokens and Gemma had 16. The same number of matches at each higher n-gram order also gave Gemma higher precision. Scores were calculated separately for each sentence against the frozen English reference using SacreBLEU 2.5.1, 13a tokenization, mixed case, exponential smoothing and effective order.

**What the learned metrics picked up.** Both gave the payslip translation a higher score. xCOMET-lite marked no error span in either output, so its score cannot identify which term influenced the result. The historical model-assisted review rated the document substitution a major error. The suggested recipient actions are a reading of the text. A human recipient did not take part in this test.

<!-- page -->

### Gemma experiments at two training budgets

**Matched small-data trial.** Two rank-8 Gemma 3 1B-IT LoRA adapters started from the same base checkpoint. Each completed 48 updates on 384 examples. One used 384 official pairs. The other used 312 official and 72 reviewed B01 training pairs, with 24 substitutions per language. On the separate historical Everyday72, mixed minus official mean COMET was +0.0124 (Polish), −0.0004 (Croatian), −0.0063 (Russian) and +0.0028 (Ukrainian). Each paired 95% interval included zero.

**Full official-text run.** A fresh Gemma 3 1B-IT adapter trained on the 300,000-pair v2 release. Query/value LoRA with rank 8 and alpha 16 updated 745,472 parameters. The run used bfloat16, zero LoRA dropout, AdamW at 0.0002, effective batch size 32 and seed 42. Randomized length buckets, microbatches of up to eight and selective activation checkpointing retained complete sequences. There were 29,955,548 input-plus-target tokens and 9,670,521 supervised English tokens including turn terminators.

One NVIDIA L4 completed 9,376 optimizer updates in 7,889.3 training seconds (2 h 11 min 29 s, 38.03 pairs/s). This timer includes transfer, forward/backward computation and optimization. Setup, checkpointing and evaluation are outside it. A matched forward/backward benchmark on the same 32 examples measured 5.48 pairs/s with single-pair microbatches and 32.17 pairs/s with up to eight. Peak allocated GPU memory rose from 2.66 to 5.65 GiB. The 5.88× speedup describes only this benchmark.

**Checkpoint evaluation.** The base model and snapshots after 10,000, 50,000 and 300,000 pairs each translated 264 messages. The reference-based results cover 248 messages per stage (48 validation, 96 official test, 32 Ukrainian FLORES, 72 everyday). Stress16 has separate decision keys. The base and adapters generated 1,056 translations in total. English-token validation loss on 48 held-out official pairs went from 3.9535 to 1.2153, 1.0964 and 0.9503. No B01 or Ukrainian examples entered the full training run.

| Language | Official test or Ukrainian FLORES COMET, base → final | Everyday COMET, base → final |
|---|---|---|
| Polish | 0.7746 → 0.8383 | 0.6743 → 0.6373 |
| Croatian | 0.7792 → 0.8653 | 0.7483 → 0.7699 |
| Russian | 0.8304 → 0.8548 | 0.7723 → 0.7452 |
| Ukrainian | 0.8280 → 0.8257 | 0.7886 → 0.7569 |

The first score column covers 32 examples per language. Ukrainian uses FLORES, a multilingual evaluation dataset. Everyday72 contains 18 messages per language. Everyday COMET rose only for Croatian, whose paired 95% interval for the change (−0.0188 to +0.0590) includes zero. The Polish and Ukrainian intervals lie below zero, while the Russian interval includes zero. These intervals resample messages from a single run. A reading of the final Croatian shift example found a time window turned into fixed endpoints. The automated reader also missed a reversed Ukrainian school-pickup role. Published scores describe these panels and this run. [All stages, BLEU, chrF++ and COMET](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/results/metrics.csv)

<!-- page -->

### Evidence register for sections 4–7

The records below distinguish the original CPU/GPU programs, public worked examples and aggregate evidence. They let readers follow a claim from dataset construction to evaluation. The institutional training archives, original model weights and individual held-out messages remain with their respective providers or project records.

| Public resource | What a reader can check |
|---|---|
| [1. Corpus preparation](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/notebooks/01_collect_and_clean.ipynb) · [Pilot builder and verifier](https://github.com/Silveroboros-dev/slavic-english-slm/tree/main/pipeline) | The notebook demonstrates source parsing, text normalization, document grouping and both sides of an overlap check. The original programs reproduce the v1 pilot from four upstream archives and verify split membership and counts. |
| [2. Scenario construction](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/notebooks/02_build_contrast_questions.ipynb) · [B01 training records](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/data/b01_contrasts.jsonl) | All 36 synthetic scenarios have two source messages, two English references, one question and opposite decisions. The notebook separates the single-version recipient display from the answer key. |
| [3. LoRA experiments](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/notebooks/03_lora_experiments.ipynb) · [Original NLLB trainer](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/training/train_nllb.py) | The notebook walks through a low-rank adjustment and loss on English answer tokens, then compares recorded experiments. The GPU program trains the original NLLB adapter on the full frozen pilot. |
| [4. Evaluation and recipient decisions](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/notebooks/evaluate_translation.ipynb) · [NLLB everyday scores](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/results/nllb_everyday.csv) | Use observed metric changes and a separate synthetic scheduling example to compare translation quality with the action a reader might take. Recipient actions in the worked example illustrate the protocol. |
| [5. Matched Gemma trial](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/results/gemma_matched_everyday.csv) · [Full-run paired intervals](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/results/everyday_comet_intervals.csv) | The small trial compares official and mixed arms at the same step. The second file estimates uncertainty on 18 everyday messages per language for the full official-text run. |
| [6. Full Gemma metrics](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/results/metrics.csv) · [Model settings and timing](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/results/gemma_run.json) | Stage-level aggregates cover BLEU, chrF++ and COMET for panels with English references. The run record identifies training parameters, precision, corpus revision and timing. Stress16 uses decision questions. |

The published notebooks contain project-authored explanations and runnable examples. The original training and inference runs used frozen datasets and model weights in Colab. Data and model terms apply separately to the respective upstream resources. [Project repository](https://github.com/Silveroboros-dev/slavic-english-slm)

<!-- page -->

## 8. Technical Risks & Mitigation Strategies

| Risk and original impact | How the risk presents | Proposed mitigation and current evidence |
|---|---|---|
| **Register mismatch · High** | Institutional training can impose formal wording on an everyday request or lose a condition. The Gemma 300,000-pair run improved official test scores while everyday scores diverged. | The original 25% subtitle/web-data mixture remains a proposal. The small matched Gemma pilot substituted 72 reviewed B01 messages for official records. Test mixtures at a larger matched budget with scenario-level holdouts. |
| **Cyrillic–Latin orthographic drift · Medium** | A multilingual encoder may separate scripts in ways that weaken transfer or fragment related words differently. | The original options were cross-script transliteration augmentation and regularization of shared subword embeddings. First compare token fertility and matched transfer performance. There is no measured representation split or completed augmentation trial. Preserve natural script and distinguish transliteration effects from translation gains. |
| **False-friend errors · High** | Cognates that have drifted in meaning may change or reverse a message, including an insulting interpretation. | The proposed experiment mines hard-negative sentence pairs and applies a targeted cross-entropy penalty. Build the section 7 contrastive test first, then compare loss variants on held-out contexts and track newly introduced errors. Attention alone offers no guarantee. |
| **Resource imbalance · Medium** | Sampling the raw pool uniformly would give the approximately 25M-row Russian UN archive far more influence than either EU language. This can bias register as well as vocabulary. | The original corpus-resampling option uses exponential smoothing (sampling exponent 0.4). The actual pilot used fixed per-language quotas of 100,000 with seeded sampling and shuffling. Compare alternative language shares on held-out data. The 0.3 tokenizer-sampling exponent in phase 2 serves a different purpose. |

### Consequences for the deployment decision

Adequacy is judged against what changes for a reader. Lost negation, a reversed boundary or a reassigned duty can matter even when fluency is high. For personal stories, participants may retell a situation without copying correspondence. Authors decide what to retain. The project owner handles corrections and removal requests, and rechecks affected evaluations. Reuse of participant examples for training requires separate consent.

If offline use improves privacy or reliability for participants, measure the practical cost on their own devices before recommending it. Compare download size, working memory, delay, battery and any new device or data-plan expense with the usual workflow. Server-side resource estimates for cloud alternatives are relevant only when documented. No environmental advantage has been measured.

<!-- page -->

## 9. Immediate action items and next steps

### Close the original acquisition and benchmarking tasks

The original checklist called for sample slices of Europarl PL/HR and Russian UNPC to assess alignment and domain distribution. The acquired pilot adds Polish JRC-Acquis and uses DGT for Croatian. The pair- and document-level provenance checks are complete. The semantic alignment gate remains open. For baseline benchmarking, the original checklist named NLLB-200 or Gemma-2-2B on FLORES-200 for PL/RU/HR with BLEU and chrF++. The paired pilot instead used held-out official text for these three and FLORES-200 for Ukrainian. Establish matched general-domain comparisons for the three anchors before closing that task.

Further controlled experiments could compare vocabulary sizes for multilingual BPE, temperatures for sampling unequal language corpora, and LoRA rank and alpha under device constraints. The existing errors already give a testable data hypothesis. Hold those choices fixed in the next comparison.

### Run a controlled test of training data

The small matched Gemma pilot provides an initial estimate of the effect of substituting 72 reviewed pairs. A stronger next comparison would initialize two runs from one saved Gemma adapter, keep one on official data and replace a prespecified share in the other with reviewed everyday messages. Match language proportions, starting optimizer state, schedule, seed, effective batch, LoRA settings and additional steps. Record supervised tokens and exposure per example. Keep Ukrainian entirely outside training.

Before selecting a final model or prompt, reserve newly contributed situations with permission. Hold out each whole scenario, its translations, variants and paraphrases. Record the author’s intended action, conditions and acceptable uncertainty first. Show each reader one blinded English version per scenario. Count decisions preserved, lost and recovered by language. Historical Everyday72 and Stress16 remain separate regression sets. A later practical comparison can include the writer’s usual method, a friend or a clarifying question.

### Completed review of the synthetic training set

Batch B01 contains 12 synthetic A/B scenarios in each of Polish, Croatian and Russian. The project initiator reviewed the Russian set and confirmed the revised approval/rejection wording for RU-05. One submitted Polish review and one Croatian review covered all 12 scenarios in their respective languages. Their answers agreed with the 24 keys per language. One reviewer per language supports the intended reading of these items. It does not estimate population-level agreement.

Both variants use the same English question, with a different gold decision. For example, printing five photos meets an “at least four” requirement but fails an “exactly four” requirement. These approved messages were training data in the small mixed Gemma trial. They are distinct from the historical 72-message test and the 16 stress cases. [How the questions were built](https://github.com/Silveroboros-dev/slavic-english-slm/blob/main/notebooks/02_build_contrast_questions.ipynb)
