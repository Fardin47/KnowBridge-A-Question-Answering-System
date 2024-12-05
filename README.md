# KnowBridge: A Question-Answering System

Introduction
This project builds an English Question Answering (QA) system using an open-source language model (DistilBERT). 
The system:
•	Fine-tunes a pre-trained (distilbert-base-uncased) QA model on the SQuAD Augmented V2 dataset from Hugging Face. (Dataset link: https://huggingface.co/datasets/christti/squad-augmented-v2)
•	Provides an interactive web interface using Gradio for answering user-provided questions based on input contexts.
•	Although English is primarily supported, this pipeline can be extended to other languages like Bangla with suitable datasets and pre-trained multilingual models.
•	Due to low GPU and Time Constraint, only 5000 samples were used from the training dataset.

Project Structure
1.	fine_tune.py: Contains the code for fine-tuning the pre-trained DistilBERT model.
2.	app.py: Implements a simple web application for question answering using Gradio.
3.	requirements.txt: Lists all required dependencies to run the project.

Workflow
1.	Fine-Tuning the Model
    o	Load the SQuAD Augmented V2 dataset.
    o	Preprocess and tokenize the data for training.
    o	Fine-tune the DistilBERT model using the preprocessed dataset.
    o	Save the fine-tuned model for deployment.
2.	Deployment
    o	Use the fine-tuned model to create a Gradio web interface.
    o	Answer questions based on user-provided contexts via an interactive GUI.

Step-by-Step Guide
  1. Setup
    1.	Clone this repository or create a new project folder. 
    -	git clone https://github.com/Fardin47/KnowBridge-A-Question-Answering-System.git
    2.  Install required dependencies: 
    -    pip install -r requirements.txt
    -    pip install 'accelerate>={ACCELERATE_MIN_VERSION}'
    3.  If you're using Windows with Python 3.8 and no GPU, the installation command:
    -	   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        For GPU support (CUDA 11.8):
    -	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  2. Fine-Tune the Model
     Run the fine-tuning script:
    -	python fine_tune.py
     This script will:
       •	Load and preprocess the dataset.
       •	Fine-tune the DistilBERT model for the QA task.
       •	Save the fine-tuned model to the ./models/fine_tuned_model directory.
  3. Run the Web Application
      To launch the Gradio web interface:
    -	python app.py
      Open the provided URL in your browser to interact with the system.

N.B: Fine-tuning generates large files, making it difficult to push to a GitHub repository. So, could not share the fine-tuned model. 
