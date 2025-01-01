# Introduction to Ollama: Your Local LLM Powerhouse

<img src="static-assets/ollama-logo.png" alt="Ollama Logo" height=60>

**Preface**

This guide will introduce you to Ollama, a powerful tool for running large language models (LLMs) LOCALLY on your own machine.

You'll learn how to install Ollama, interact with its models, and explore the benefits of having LLMs at your fingertips without relying on external APIs or internet connectivity.

**Goals**

By the end of this guide, you should:

- Have Ollama installed and running on your system.
- Understand the advantages of using local LLMs.
- Be able to download and run various Ollama models.
- Know how to use Ollama for offline translation and as a personal information assistant.

**Main Resources**

- **Required Software:**
  - Python 3.9+ (have this locally installed)
  - Ollama CLI ([https://ollama.ai](https://ollama.ai))
- **Core Services:**
  - Ollama Server
  - Ollama Model Library
- **Development Tools:**
  - Terminal or Command Prompt
  - Project IDX or IDE of choice

**Tasks**

- **Ollama Installation:**
  - Download and install the Ollama CLI for your operating system.
- **Model Exploration:**
  - Download various models from the Ollama library (e.g., `llama3.2`, `gemma-2-jpn-translate`).
  - Run models and experiment with different prompts.
- **Use Case 1: Offline Translation:**
  - Use a translation model for offline language translation.
- **Use Case 2: Personal Information Assistant:**
  - Create a simple information assistant using a local model and your own data.

## Instructions

- **Install Ollama:**
  - Follow the installation instructions on the Ollama GitHub page: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
- **Run the Ollama Server:**
  - Open a terminal and run `ollama serve`.

**Model Usage**

- **Download a model:**
  - Use the `ollama pull` command (e.g., `ollama pull llama3.2`).
- **Run a model:**
  - depending on OS you might need `ollama serve` first
  - Use the `ollama run` command (e.g., `ollama run llama3.2`).
- **Interact with the model:**
  - Type your prompts and press Enter to get responses.

## Use Case Demos

**1. Offline Translation:**

- **Download a translation model:** `ollama pull 7shi/gemma-2-jpn-translate:2b-instruct-q8_0`
- **Run the model:** `ollama run 7shi/gemma-2-jpn-translate:2b-instruct-q8_0`
- **Translate text:** Enter Japanese text and get the English translation.

**2. Personal Information Assistant:**

- see `local-assistant.py` for demo
- we are running a secret assistant for a superhero that's undercover in the batcave

## Further Resources

**Documentation:**

- Ollama Documentation: [https://ollama.ai/docs](https://ollama.ai/docs)
- Ollama GitHub Repository: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

**Helpful Tips**

- **Choose the right model:** Select models appropriate for your task and hardware capabilities.
- **Experiment with prompts:** Craft clear and specific prompts to get the best results.
- **Explore different models:** Try various models to find the ones that suit your needs.
- **Consider quantization:** Use quantized models to reduce size and improve performance.
- **Enjoy the benefits of local LLMs:** Experience the speed, privacy, and offline accessibility of Ollama.
