from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForTokenClassification
from nltk import word_tokenize
import torch
import requests
from transformers import DistilBertForSequenceClassification

# Load the tokenizer and model
model_path = 'best_model_Latest'  # Replace with the actual path to your trained model
tokenizer_path = 'best_model_tokenizer_Latest'  # Replace with the actual path to your saved tokenizer

tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
# FastAPI app
app = FastAPI()

class TextRequest(BaseModel):
    text: str

class PredictLabel(BaseModel):
    label: int   

@app.post("/predict_labels", response_model=PredictLabel)
def predict_label(request: TextRequest):
    try:
        # Tokenize the input text
        tokens = tokenizer.encode(request.text, add_special_tokens=True, padding=True, truncation=True)

        # Converting tokens to tensors
        input_ids = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
        attention_mask = torch.ones(input_ids.shape)

        # Make the prediction for the segment
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        predicted_label = int(torch.argmax(outputs.logits).item())
   

        return {"label":predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))