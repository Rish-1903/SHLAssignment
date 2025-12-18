# SHLAssignment

## Grammar Scoring Engine for Spoken Data
This repository contains a Multimodal Speech-to-Text and NLP Pipeline designed to automatically assess the grammatical proficiency of spoken audio samples. The engine outputs a continuous Mean Opinion Score (MOS) ranging from 0 to 5 based on a specific linguistic rubric.

## 📌 Problem Description

The goal of this project is to develop a model that evaluates 45-60 second audio recordings of spoken English. Unlike traditional audio classification, this task requires understanding complex sentence structures, syntax, and grammatical accuracy.

## Grammar Scoring Rubric

The model is trained to predict scores based on the following criteria:
Score	Description
1	    Struggles with proper sentence structure; limited control over simple patterns.
2	    Limited understanding of syntax; consistent basic mistakes; incomplete sentences.
3	    Decent grasp of structure but makes errors in grammatical structure or syntax.
4	    Strong understanding; consistent good control; minor errors do not lead to misunderstanding.
5	    High grammatical accuracy; adept control of complex language structures; self-corrects.

## 🏗️ Pipeline Architecture

The solution uses a two-stage approach to bridge the gap between acoustic signals and linguistic evaluation:

    Speech-to-Text (STT) Layer: Utilizes OpenAI Whisper (Base) to convert .wav files into high-quality text transcripts, preserving punctuation and flow.

    Linguistic Regression Layer: A RoBERTa-base transformer model fine-tuned for regression. RoBERTa is chosen for its superior ability to understand context and syntactic relationships.
Install the required libraries:
    
    pip install openai-whisper transformers datasets librosa torch tqdm scikit-learn
    
## Output Result
    0.504 at Kaggle 

