## Cultural AI Assistant and Story Telling agent

## Overview  
My project is an AI assistant trained on Ugandan legends, sayings, and short stories.  
It allows users to explore traditional stories, learn moral lessons, and understand sayings with translations and cultural context.  

## Why this project  
Uganda has a rich oral tradition, but many legends and sayings risk being forgotten. This is because of the new wave of technology, which leaves no time for parents to sit down with their children to pass on some of these traditions. Given the amount of time I have taken to compile these stories, it highlights the need for a hub that makes them available. The agent can be used to better understand them, simplify them for students who are having a hard time, and provide knowledge for anyone just curious.

This project aims to preserve cultural wisdom in a digital form, making it accessible for future generations, students, and anyone interested in Luganda heritage.  

## Features  
- Retrieve and narrate the legends and short stories.  
- Explain lessons and morals from stories.  
- Provide Luganda sayings, English translations, and cultural categories.
- Be able to simplify or complicate vocabulary based on the level of the user(kid, teenager, adult, etc.)   
- Search across stories and sayings using embeddings + retrieval.  
- Simple chatbot interface to interact with the knowledge base.  

## Tech Stack  
- Python
- LangChain for retrieval  
- OpenAI / HuggingFace embeddings  
- Streamlit (for user interface)  
- JSON datasets (legends, short stories, sayings)  

## Dataset Format  

### Legends & Short Stories  
```json
[
  {
    "title": "Kintu and Nambi",
    "content": "Once upon a time...",
    "lesson": "True worth is shown through effort and cleverness..."
  }
]
