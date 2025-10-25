# OmniFlowAI – A Unified Multi-Modal Intelligence Pipeline

## Overview

**OmniFlowAI** is an advanced **multi-modal AI system** that processes and transforms information across **text, image, audio, video, and tabular formats** — all within a single intelligent workflow.

It integrates **Retrieval-Augmented Generation (RAG)**, **fine-tuned models**, and **blending techniques** to deliver accurate, context-aware, and format-flexible outputs.  
From converting PDFs to narrated summaries to turning CSVs into dynamic video insights, OmniFlowAI provides a unified AI agent for multimodal automation.

---

## Project Structure


1.AI_challenge.ipynb # Jupyter Notebook: training, fine-tuning, experiments
2.agent_to_run.py # Automated AI agent for multimodal processing
3.data/ # Input files (PDF, CSV, images, PPT, etc.)
4.outputs/ # Generated results (text/audio/video)
5.requirements.txt # Python dependencies
6.README.md # Project documentation

---

## Key Features

**Multi-Modal Input Handling** — Works with PDFs, images, PPTs, CSVs, and structured templates.  
**RAG-Based Context Retrieval** — Retrieves grounded and factual information before generation.  
**Fine-Tuned + Blended Models** — Combines fine-tuned models with post-filtering for higher accuracy.  
**Cross-Format Conversion** — Converts inputs into multiple outputs like text, audio, and video.  
**Automation via Agent** — The `agent_to_run.py` file automates the full end-to-end pipeline.  

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/OmniFlowAI.git
   cd OmniFlowAI
   
2. Create a virtual environment
   
       python -m venv venv
       source venv/bin/activate      # Mac/Linux
       venv\Scripts\activate         # Windows

3. Install dependencies

       pip install -r requirements.txt


Usage

Run Notebook (for Training & Experiments)

    jupyter notebook AI_challenge.ipynb

Use this notebook to:

    Explore multimodal data processing

    Fine-tune and blend models

    Run and visualize RAG pipelines

Run Agent (for Full Automation)

     python agent_to_run.py

The agent:

    Accepts multimodal inputs (PDFs, CSVs, images, PPTs, etc.)

    Retrieves relevant context via RAG

    Generates and converts results into multiple formats

    Saves final outputs to the outputs/ folder

Example Workflow:

  Input: PDF, image, or CSV file
  Pipeline: Preprocessing → RAG Retrieval → Fine-Tuned Inference → Format Conversion
  Output:

    📄 Text Summary (.txt)

    🔊 Audio Narration (.mp3 / .wav)

    🎬 Captioned Video (.mp4)

Example Applications:

    Summarize reports or research papers into voice-narrated summaries

    Convert tabular datasets into visual or narrated presentations

    Create AI-generated videos from PowerPoint decks

    Generate contextual responses grounded in local knowledge bases
