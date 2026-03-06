## SEO Specialist ChatBot 

![SEO Expert Chatbot Diagram](assets/SEO%20Expert%20Chatbot%20App%20Diagram.jpg)
SEO Expert Bot is a Retrieval Augmented Generation (RAG) chatbot designed to deliver expert-level SEO strategy recommendations using a curated knowledge base of approximately 1000 pages of SEO documentation (internally used REAL data from a SEO agency). 

The system integrates external documents stored in Google Docs, processes them into vector embeddings, and stores them in a Chroma vector database for efficient semantic retrieval. 

When a user submits a query, the system retrieves relevant contextual information and generates accurate, context-grounded responses using a frontier LLM. The application is deployed with a streaming Gradio interface, providing real-time conversational interaction and demonstrating a production-style RAG architecture.
Made with LangChain & RAG and Google Docs API as an external data resource.

## Demo Video:
https://drive.google.com/file/d/1Q850jOtNC7sA9GbBXdDGhKtIgm8S0e9f/view?usp=sharing 

### Technical Features:
1. Full RAG system using LangChain, OpenAI Embeddings, and Chroma DB
2. Vector database built from ~1000 pages of SEO documentation used internally from an actual SEO agency (Fully permitted for the data use)
3. External document ingestion using Google Docs API and Google Cloud Service Account
4. Semantic search pipeline for context-aware retrieval
5. Context-grounded response generation using frontier LLM (GPT-5-mini)
6. Persistent vector store for efficient inference performance
7. Streaming chat interface deployed using Gradio
8. Modular and scalable architecture

Live App on Hugging Face: https://huggingface.co/spaces/MightyOctopus/seo-expert-chatbot 

