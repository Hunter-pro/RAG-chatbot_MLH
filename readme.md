# Local AI Chat Interface

A Streamlit-based chat interface that works with LM Studio for local AI interactions. This project provides a ChatGPT-like experience using your own locally hosted language models.

## 🌟 Features

- Clean, intuitive chat interface
- Real-time streaming responses
- Local model support via LM Studio
- Message history persistence during session
- Error handling for connection issues

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- [LM Studio](https://lmstudio.ai/) installed on your machine
- A compatible language model loaded in LM Studio

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Setup

1. Start LM Studio and load your preferred language model
2. Ensure LM Studio's server is running on port 1234 (default port)
3. Run the Streamlit app:
   ```bash
   streamlit run pages/😊AI_Chat.py
   ```

## 🔧 Usage

1. Open your browser and navigate to the Streamlit app (typically `http://localhost:8501`)
2. Start chatting with your local AI model
3. The chat history will be maintained during your session

## ⚠️ Important Notes

- LM Studio must be running with a loaded model before starting the chat interface
- The server runs on `http://localhost:1234` by default
- No API key is required as this uses local models

## 🛠️ Troubleshooting

If you encounter issues:

1. Verify LM Studio is running and a model is loaded
2. Check if the LM Studio server is running on port 1234
3. Look for error messages in the Streamlit interface
4. Ensure all dependencies are correctly installed

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
Built with Streamlit and LM Studio for local AI interactions 🤖
