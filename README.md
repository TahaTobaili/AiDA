# AiDA: The Youtube Chat Assistant

Welcome to AiDA, a prototype built in 1 day for 42-London Hackathon that was focused on Redefining Education with AI. AiDA enables you to chat with your youtube videos as they play in real-time. It also tries to classify your intent, if you are looking for a summary it sketches a quick chart to explain the main points of the video. The goal is to bring videos to life to improve children's learning experience on youtube. 

[![Watch the video](https://github.com/TahaTobaili/AiDA/blob/main/Thumbnail.png)](https://www.youtube.com/watch?v=-rsWLIZn9Wo)

**P.S: _AiDA got 1st place winning the 42 Hackathon!_**

## Requirements

AiDA is web based app built using **Python** and **streamlit**, it utilizes **Langchain** for Retrieval Augmented Generation (RAG) chatbot, vectorized document storage with **OpenAI** and **Chroma**, GPT/Claude models for analysing quering and chatting, GPT to generate mermaid chart and **streamlit mermaid library** to sketch the chart.


## Running AiDA

To get started with AiDA, ensure you have Python 3.6+ installed. Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourgithub/AiDA.git
cd AiDA
streamlit run app.py
```

## Contributing

Any contributions you make are **greatly appreciated**. If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue and give it the correct labels.
