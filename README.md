<div align="center">
  <img src="https://img.shields.io/badge/Framework-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface" alt="Framework Badge">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python" alt="Python Badge">
  <img src="https://img.shields.io/badge/Paradigm-Curriculum--Aligned%20AI-green?style=for-the-badge&logo=googleclassroom" alt="Curriculum-Aligned AI Badge">
  <img src="https://img.shields.io/github/stars/cotix-ai/LearnLM-nano?style=for-the-badge&color=gold" alt="Stars Badge">
</div>

<br>

<h1 align="center">
  LearnLM-nano: An Open-Source LLM Fine-Tuned for Education
</h1>

<p align="center">
  <i>Powering the next generation of AI-driven educational tools with a model that understands how students learn.</i>
</p>

<br>

>[!NOTE]
> LearnLM-nano is designed to be a lightweight, efficient, and accessible model, making it ideal for deployment in educational applications, classroom tools, and research platforms without requiring extensive computational resources.

## 🌟 Table of Contents

-   [🌟 Table of Contents](#-table-of-contents)
-   [✨ Introduction](#-introduction)
-   [💡 Core Philosophy: From General Knowledge to Structured Pedagogy](#-core-philosophy-from-general-knowledge-to-structured-pedagogy)
-   [🧠 The Dataset Core: The K-12 Curriculum Corpus](#-the-dataset-core-the-k-12-curriculum-corpus)
-   [🧩 Fine-Tuning Methodology](#-fine-tuning-methodology)
-   [🚀 Why LearnLM-nano? Unique Advantages in Education](#-why-learnlm-nano-unique-advantages-in-education)
-   [🔧 Getting Started & Usage](#-getting-started--usage)
-   [🤝 Contribution](#-contribution)

<br>

---

## ✨ Introduction

This project introduces **LearnLM-nano**, a compact and efficient language model specifically fine-tuned to excel in the K-12 educational domain.

While general-purpose large language models (LLMs) possess vast knowledge, they often lack the pedagogical nuance required for effective teaching. They can provide correct answers, but may fail to explain concepts in an age-appropriate, step-by-step manner that fosters genuine understanding.

**LearnLM-nano** addresses this gap by fine-tuning a robust base model on a high-quality, curated dataset of K-12 educational materials. It learns not just the "what" (the answer) but the "how" and "why" (the explanation and methodology). This makes it a specialized tool for creating safer, more effective, and context-aware AI learning assistants.

<br>

---

## 💡 Core Philosophy: From General Knowledge to Structured Pedagogy

**LearnLM-nano represents a fundamental shift from training on the open web to curating knowledge from structured curricula.** We believe that for an AI to be a truly effective educational partner, it must be aligned with the same principles that guide human educators: scaffolding knowledge, providing clear explanations, and encouraging critical thinking.

> "True AI-powered education isn't about giving answers; it's about guiding students through the process of discovery."

Our approach moves beyond simple question-answering. We aim to imbue the model with a sense of pedagogical structure, enabling it to break down complex problems, explain concepts with relevant analogies, and adapt its responses to a student's learning level.

<br>

---

## 🧠 The Dataset Core: The K-12 Curriculum Corpus

The **K-12 Curriculum Corpus** is the **bedrock** of the LearnLM-nano project and serves as the **"Single Source of Truth"** for its educational alignment. This dataset is meticulously curated to reflect the structure and content of standard K-12 learning paths.

**Corpus Composition:**
The model is trained on a diverse set of educational data formats, including:
1.  **Instructional Texts:** Explanations of core concepts in subjects like Math, Science, and History, written in clear, accessible language.
2.  **Question-Answer Pairs:** A vast collection of homework problems, quiz questions, and their canonical answers.
3.  **Step-by-Step Solutions:** Detailed, methodical walkthroughs for solving complex problems, teaching the process rather than just the result.
4.  **Socratic Dialogues:** Simulated teacher-student interactions that guide a student to an answer through a series of questions.

By training on this corpus, LearnLM-nano learns the patterns of effective teaching, making its outputs not only accurate but also instructionally valuable.

<br>

---

## 🧩 Fine-Tuning Methodology

The intelligence of LearnLM-nano is cultivated through a specialized fine-tuning process designed to maximize its educational capabilities.

1.  **Base Model Selection:** We begin with a high-performing, open-source "nano" model (e.g., a smaller version of Llama, Gemma, or Mistral) renowned for its strong reasoning and language capabilities.
2.  **Instructional Formatting:** The entire K-12 Corpus is structured into a consistent instruction-following format (e.g., `[INST] User Question [\/INST] Model's Explanatory Answer`). This teaches the model to behave as a helpful, responsive educational assistant.
3.  **Supervised Fine-Tuning (SFT):** The model undergoes SFT on the formatted dataset. During this phase, it adjusts its weights to minimize the difference between its generated responses and the high-quality, pedagogically sound examples from the corpus.
4.  **Safety and Alignment:** The process includes rigorous filtering of the dataset to ensure all content is age-appropriate and aligned with educational standards, creating a model that educators can trust.

<br>

---

## 🚀 Why LearnLM-nano? Unique Advantages in Education

While larger models may know more facts, LearnLM-nano offers distinct, critical advantages for educational applications.

*   **Pedagogical Alignment:** The model excels at providing step-by-step explanations, using analogies, and breaking down complex topics, mimicking the methods of an effective teacher.
*   **Domain-Specific Accuracy:** By focusing on the K-12 curriculum, the model provides more reliable and contextually appropriate answers for academic subjects, reducing subject-matter "hallucinations."
*   **Enhanced Safety:** Trained on a closed, curated corpus of educational material, LearnLM-nano is significantly less likely to generate inappropriate or irrelevant content, making it safer for use by children.
*   **Efficiency and Accessibility:** Its "nano" size allows it to run on less powerful hardware, including local devices or school servers, lowering the barrier to entry for developing and deploying AI-powered educational tools.
*   **A Foundation for Innovation:** LearnLM-nano provides a solid, open-source foundation for developers to build specialized tools like personalized tutors, intelligent homework assistants, and creative learning games.

<br>

---

## 🔧 Getting Started & Usage

Getting started with LearnLM-nano is simple using the Hugging Face `transformers` library.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "learnlm-nano"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = """[INST]
I need help with a math problem. Can you explain how to find the area of a rectangle that is 8 cm long and 5 cm wide? Please explain it step-by-step.
[/INST]"""

# Generate a response
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
**Expected Output:**
```
[INST]
I need help with a math problem. Can you explain how to find the area of a rectangle that is 8 cm long and 5 cm wide? Please explain it step-by-step.
[/INST]
Of course! I'd be happy to help you with that. Finding the area of a rectangle is straightforward. Here’s how you can do it step-by-step:

Step 1: Understand the Formula
The formula to find the area of a rectangle is: Area = Length × Width.

Step 2: Identify Your Values
In your problem, the Length is 8 cm and the Width is 5 cm.

Step 3: Apply the Formula
Now, we just need to multiply the length and the width together:
Area = 8 cm × 5 cm

Step 4: Calculate the Final Answer
8 multiplied by 5 is 40. Since we multiplied centimeters by centimeters, our unit is square centimeters (cm²).

So, the area of the rectangle is 40 cm².
```

---

## 🤝 Contribution

We welcome and encourage contributions to this project! Whether it's improving the dataset, suggesting fine-tuning enhancements, or finding bugs, please feel free to submit a Pull Request or create an Issue.
