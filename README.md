# Content-Summarization-from-YouTube-and-Websites-Using-AI
# 📰 Content Summarization from YouTube and Websites Using AI

## 🚀 Overview

This AI-powered application extracts and summarizes content from YouTube videos and websites using **LangChain** and **Groq's Gemma-7B-IT model**. With a simple Streamlit interface, users can input a URL and receive an AI-generated summary of the content, making information consumption more efficient and streamlined.

## ✨ Features

- 🎥 **YouTube Summarization**: Extracts transcripts from YouTube videos and provides concise summaries.
- 🌐 **Webpage Summarization**: Fetches text from websites and condenses it into a digestible format.
- 🤖 **AI-Powered Summaries**: Uses LangChain with Groq's Gemma-7B-IT model to generate high-quality summaries.
- 🛠 **User-Friendly Interface**: Built with Streamlit for an interactive and seamless experience.

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **LLM Model**: Groq's **Gemma-7B-IT**
- **AI Framework**: LangChain



## 🔧 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/krishshah9944/Content-Summarization-from-YouTube-and-Websites-Using-AI.git
cd Content-Summarization-from-YouTube-and-Websites-Using-AI
```

### 2️⃣ Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up API Key

Create a `.env` file in the root directory and add your Groq API key:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Alternatively, you can enter the API key directly in the Streamlit sidebar.

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

## 📌 Usage

1️⃣ **Enter your Groq API key** in the Streamlit sidebar.

2️⃣ **Provide a URL** (YouTube video or website) in the input field.

3️⃣ Click on **"Summarize the Content from YT or Website"**.

4️⃣ Wait for the AI to generate a summary of the content.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests for improvements and bug fixes.

## 📧 Contact

For inquiries, please reach out via:

- **LinkedIn**: [Krish Shah](https://www.linkedin.com/in/krishshah9944/)
- **Email**: [krishshah9944@gmail.com](mailto\:krishshah9944@gmail.com)

