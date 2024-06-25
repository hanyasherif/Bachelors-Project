from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)

# Load your trained model and tokenizer
model_path = "saved_model"  # Update with the path to your trained model directory
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate_prompt", methods=["POST"])
def generate_prompt():
    if request.method == "POST":
        input_text = request.form["input_text"]

        # Tokenize input text
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        # Generate text with sampling-based generation
        output = model.generate(input_ids, max_length=1000, num_return_sequences=1, temperature=0.7, do_sample=True)

        # Decode generated text
        generated_prompt = tokenizer.decode(output[0], skip_special_tokens=True)  # Take the first item from output

        # Replace apostrophe character with the desired character
        generated_prompt = generated_prompt.replace("'", "’")
        return render_template("index.html", input_text=input_text, generated_prompt=generated_prompt)

if __name__ == "__main__":
    app.run(debug=True)





