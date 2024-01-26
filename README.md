# Chatbot Project

## Description
This chatbot is an interactive application designed for natural language processing and user interaction. It leverages a virtual environment to manage dependencies, ensuring consistent runtime behavior.

## Features
- **Natural Language Processing**: Understands and processes user input.
- **Emotion Detection**: Utilizes Hugging Face's pipeline for sentiment analysis.
- **Customizable Responses**: Based on predefined patterns in `intents.json`.
- **Virtual Environment**: All dependencies contained for ease of setup.

## Installation

### Prerequisites
- Python installed on your system.
- Git for cloning the repository.

### Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Lungu1107/chatbot_project.git
   cd chatbot_project
   
## Activate the Virtual Environment

Before running the chatbot, make sure to activate the virtual environment which contains all the necessary dependencies.

- **Navigate to the Virtual Environment Directory**:
  This directory is usually named `venv` or `env` and should be located in the root of your project directory.

- **Activate the Virtual Environment**:
  - On Windows, use:
    ```bash
    .\venv\Scripts\activate
    ```
  - On Unix or MacOS, use:
    ```bash
    source venv/bin/activate
    ```

  Replace `venv` with the actual name of your virtual environment directory, if it's different.

## Run the Chatbot

Once the virtual environment is activated and all dependencies are in place, you can start the chatbot:

```bash
python chatbot.py
