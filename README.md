# MetaSense
### Decoding Cognitive Metaphors in the Qur'an through Transformer-based Sensory Mode Classification

**MetaSense** is an open research framework for studying **cognitive metaphors** in Qur'anic discourse using modern Arabic Transformer models. The project investigates how pre-trained language models represent metaphorical meaning by predicting the **sensory mode** (النمط الحسي) underlying Qur'anic metaphors.

Unlike traditional metaphor identification tasks, MetaSense focuses on the **conceptual source domain** of metaphors from the perspective of **Conceptual Metaphor Theory (CMT)**, where abstract concepts are expressed through concrete sensory experiences.

---

## Research Motivation

Human cognition frequently understands abstract concepts through sensory experience. In the Qur'an, numerous metaphors evoke perception through vision, movement, touch, hearing, taste, and other experiential domains.

Examples include:

- **Visual (بصري):** light, darkness, blindness
- **Kinetic (حركي):** walking, falling, ascending
- **Tactile (لمسي):** hardness, softness, touching
- **Auditory (سمعي):** hearing, calling
- **Gustatory (ذوقي):** tasting consequences (*فذاقت وبال أمرها*)
- **Abstract Cognitive (إدراكي):** guidance, faith, certainty
- **Composite (مركب):** metaphors combining multiple sensory domains

MetaSense investigates whether Arabic Transformer models can automatically recognize these conceptual sensory domains from Qur'anic text.

---

## Research Questions

The project addresses several research questions:

- Which Arabic Transformer architecture best captures sensory metaphors in Classical Arabic?
- Do models trained on Classical Arabic outperform dialect-oriented models?
- How transferable are contextual language representations to cognitive metaphor analysis?
- Which sensory categories remain challenging for modern language models?

---

## Dataset

Each instance consists of a Qur'anic metaphor annotated with its corresponding **Sensory Mode**.

Typical fields include:

- Ayah text
- Metaphorical segment
- Sensory Mode label
- Surah
- Ayah number

### Target Labels

| Label | Description |
|--------|-------------|
| `visual` | Vision and imagery |
| `kinetic` | Motion and movement |
| `tactile` | Touch and physical sensation |
| `auditory` | Sound and hearing |
| `gustatory` | Taste |
| `abstract_cognitive` | Pure conceptual representation |
| `composite` | Multiple sensory domains |

---

## Models

MetaSense is designed as a benchmarking framework for Arabic language models, including:

- CAMeLBERT-CA
- ARBERT
- MARBERTv2
- AraBERT
- Other Hugging Face compatible Arabic Transformers

Additional models can be integrated with minimal code changes.

---

## Experimental Design

The benchmark follows a reproducible evaluation protocol. 

To explore the dataset and replicate the experiments, you can access the interactive notebook here: [MetaSense Replication Notebook](https://drive.google.com/file/d/1zv3eiaAzrtafQSQtqrS9zDEflvELbxEL/view?usp=sharing).

- Multi-class classification
- Three random seeds
- Stratified data split
- Macro-F1
- Weighted-F1
- Accuracy
- Precision
- Recall

Statistical significance can be evaluated using tests such as:

- McNemar's Test
- Wilcoxon Signed-Rank Test

---



## Research Areas

MetaSense lies at the intersection of:

- Computational Linguistics
- Arabic Natural Language Processing (Arabic NLP)
- Qur'anic Arabic
- Cognitive Linguistics
- Conceptual Metaphor Theory
- Semantic Representation Learning
- Transformer Language Models
- Digital Humanities

---



## Acknowledgment

MetaSense aims to facilitate reproducible research on Qur'anic metaphor understanding and to provide a benchmark for evaluating Arabic Transformer models on cognitively grounded semantic tasks.

The project welcomes contributions from researchers in:

- Arabic NLP
- Computational Linguistics
- Qur'anic Studies
- Corpus Linguistics
- Cognitive Semantics
- Digital Humanities

---

## Keywords

Qur'an • Arabic NLP • Computational Linguistics • Cognitive Linguistics • Conceptual Metaphor Theory • Sensory Mode Classification • Transformer Models • Classical Arabic • Semantic Representation • Qur'anic Metaphors
