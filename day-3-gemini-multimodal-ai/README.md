# Gemini on Google AI Studio

**Preface**

This guide will walk you through using Google AI Studio with Gemini. You'll learn how to set up and interact with this powerful AI model, understand its capabilities and learn best practices for using it effectively.

**Goals**

By the end of this guide, you should:

- Have working access to Gemini in Google AI Studio
- Understand Gemini's capabilities
- Be able to make basic API calls
- Test out one multimodal aspect of Gemini

**Main Resources**

- **Required Accounts:**
  - Google Cloud API Key
- **Core Services:**
  - Google AI Studio
  - Gemini API
  - `google-generativeai` Python library
- **Development Tools:**
  - Python 3.9+
  - Google AI Studio environment

**Tasks**

- **Google AI Studio Setup:**
  - Set up authentication (API Key) on upper top left button
- **Basic Integration:**
  - Install the `google-generativeai` package
  - Configure API key
- **Gemini Exploration:**
  - Understand capabilities
  - Test multimodal abilities

## Instructions

- **API Key Management:**
  - Create an API key for your project.
  - Securely store your API key (e.g., using environment variables or a secrets manager).
  - we are using .env file here

**Environment Setup**

- **Code:**
  - Use the jupyter notebook in IDX or colab or locally
  - ensure requirements.txt is installed
  - Replace "YOUR_API_KEY" with your actual API key.
  - Run the code and experiment with different prompts.

## Further Resources

**Documentation:**

Google AI Studio: [https://ai.google.dev/gemini-api/docs/ai-studio-quickstart]  
Gemini API: [https://ai.google.dev/]  
google-generativeai library: [https://pypi.org/project/google-generativeai/]

**Learning Materials:**
Gemini Prompt Engineering Guide: [https://ai.google.dev/gemini-api/docs/prompting-intro]  
Imagen3: [https://ai.google.dev/gemini-api/docs/imagen]

**Helpful Tips**

- Keep your API key secure. Do not commit it into any public respository.
- Monitor your API usage to avoid unexpected costs.
- Test with small requests first to understand the model's behavior.
- Document error messages for troubleshooting.
