import gradio as gr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load the fine-tuned model and tokenizer
model_name_or_path = "./models/fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)

# Define a pipeline for question answering
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Define a function to handle user input
def answer_question(context, question):
    response = qa_pipeline({"context": context, "question": question})
    return response["answer"]

# Gradio interface
interface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="Context", lines=10, placeholder="Enter context here..."),
        gr.Textbox(label="Question", placeholder="Enter your question here..."),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Question Answering System",
    description="Enter a context and a question to get an answer.",
)

# Launch the app
interface.launch()
