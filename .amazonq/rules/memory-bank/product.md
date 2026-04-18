# Anubis — Product Overview

## Purpose
Anubis is an AI Personality Reconstruction system that creates digital avatars of real people for automated social messaging response, self-improvement, bioinformatics, and entertainment. It is described as "A Celebration of the Best of a Person" — a LangGraph agent server for afterlife systems avatar creation.

## Core Value Proposition
- Reconstruct a person's personality, voice, and identity from multi-modal data (text, audio, video, images, social media)
- Enable authentic, personalized AI responses that reflect the real person's communication style, beliefs, and emotional patterns
- Provide a persistent, evolving digital identity that learns and remembers over time

## Key Features

### Identity & Memory
- Dynamic system prompt construction from stored identity documents, memories, and direct quotes
- Episodic memory creation and recall via LangGraph store (vector-indexed)
- User and assistant identity namespaces: `(creator_id, assistant_id, "identity")`, `(user_id, assistant_id, "memory")`, `(creator_id, assistant_id, "quote")`
- Reference image description integration into identity context

### Conversation & Reasoning
- Multi-step reasoning loop: `load_consciousness → think → process_thoughts → respond`
- Internal thought processing with tool-use before final response
- Terms of service and privacy policy content moderation
- Message trimming and token management

### Data Ingestion & Processing
- Multi-modal data support: YouTube videos, audio files, images, PDFs, social media exports
- Vectorstore indexing pipeline for documents
- Adapter training data preparation from conversation data
- Ground truth baseline storage for authenticity evaluation

### Psychological Analysis
- OCEAN (Big Five) personality analysis
- Plutchik emotional wheel emotion tracking
- MBTI personality type analysis
- Psycho-analysis prompt suite

### API & Integration
- FastAPI web application with LangGraph HTTP integration
- Auth0 + Firebase + Supabase authentication
- Slack SDK integration
- Stripe subscription/metering
- Prometheus metrics + Grafana dashboards

## Target Users
- Individuals who want to preserve and interact with a digital version of themselves or loved ones
- Researchers in bioinformatics and personality modeling
- Developers building personalized AI assistant products
- Entertainment and media applications requiring authentic character simulation

## Data Sources Supported
- YouTube transcripts (yt-dlp)
- Social media exports (tweets, Facebook, ChatGPT/Claude/Grok chat history)
- Audio/video recordings
- Text documents, PDFs, biographies
- Handwritten notes and Q&A datasets
- LLM conversation histories
