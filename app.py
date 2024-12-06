from flask import Flask, render_template, request
from transformers import pipeline
import pdfplumber 
import os

app = Flask(__name__)

# Load a more accurate question-answering model
qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

@app.route("/", methods=["GET", "POST"])
#define function
def index():
    if request.method == "POST":
        uploaded_file = request.files["pdf"]
        question = request.form["question"]

        try:
            # Save the uploaded PDF temporarily
            pdf_path = "temp.pdf"
            uploaded_file.save(pdf_path)

            # Extract text from the PDF using PDFPlumber
            pdf_text = extract_text_from_pdf(pdf_path)

            # Use the question-answering model to get the answer
            answer = get_answer_from_pdf(question, pdf_text)

            # Delete the temporary PDF file
            os.remove(pdf_path)
            
            if not answer:
                answer = "No answer found."

            return render_template("index.html", answer=answer)
        except Exception as e:
            error_message = str(e)
            return render_template("index.html", error=error_message)

    return render_template("index.html", answer=None, error=None)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_answer_from_pdf(question, text):
    result = qa_model(question=question, context=text)
    return result["answer"]

if __name__ == "__main__":
    app.run(debug=True)

