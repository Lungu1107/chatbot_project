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
      # This command will launch the chatbot interface where you can interact with it.

## Customizing the Chatbot

To tailor the chatbot to specific needs or add new functionalities, follow these steps:

### Modify Intents

1. **Edit the `intents.json` File**: This file contains the various patterns and responses that the chatbot uses to interact with users. 
   - To add a new intent, include a new tag with its associated patterns and responses.
   - To modify an existing intent, update the patterns or responses under the respective tag.

### Retrain the Model

2. **Retrain the Chatbot Model**:
   - After updating `intents.json`, run the `new.py` script to retrain the model with the new data. This ensures that your changes are reflected in the chatbot's behavior.
   ```bash
   python new.py
   # The script will process the updated intents and retrain the neural network model, saving the new model for the chatbot to use.

## Test Your Changes

After customizing and retraining your chatbot, it's important to thoroughly test the changes to ensure the chatbot behaves as expected.

### Steps for Testing

1. **Run the Chatbot**:
   - Start the chatbot using the command:
     ```bash
     python chatbot.py
     ```
   - This will launch the chatbot interface.

2. **Interact with the Chatbot**:
   - Try out different queries that align with the patterns you've added or modified in the `intents.json` file.
   - Pay attention to how the chatbot responds to ensure it's picking up the right intents and providing appropriate responses.

3. **Check for Emotion-based Responses**:
   - If you've implemented emotion-specific responses, test these by inputting text that triggers various emotions.
   - Verify that the chatbot's responses align with the emotion detected in the input.

4. **Validate Functionality**:
   - Ensure that all functionalities of the chatbot are working correctly.
   - If you've made any additional changes or enhancements, test these thoroughly to confirm they integrate well with existing features.

### Debugging Issues

- If you encounter unexpected behavior or responses, revisit the `intents.json` and scripts (`new.py` and `chatbot.py`) to troubleshoot and refine the logic or training data.
- For debugging complex issues, you might need to add print statements or use a debugger to trace the execution flow and inspect variables.

## Keeping the Chatbot Updated

Regularly update and improve your chatbot based on user interactions and feedback. This iterative process helps in refining the chatbot's accuracy and user experience.

## Contributing

Your contributions to enhance and improve this chatbot are welcome. Whether it's adding new features, fixing bugs, or improving documentation, your input is valuable:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -am 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Create a new Pull Request.

## Acknowledgements

- A special thanks to the open-source community and contributors who have made tools like TensorFlow, NLTK, and Hugging Face's Transformers available for public use.
