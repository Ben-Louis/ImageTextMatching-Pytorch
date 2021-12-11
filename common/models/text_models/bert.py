import os
import torch
from transformers import BertTokenizer, BertModel
CURR_DIR = os.path.abspath(__file__).rsplit(os.sep, 1)[0]

def bert_inference_gen(device="cpu"):
    if os.path.exists(os.path.join(CURR_DIR, "bert-base-multilingual-cased")):
        tokenizer = BertTokenizer.from_pretrained(os.path.join(CURR_DIR, "bert-base-multilingual-cased"))
        feature_extractor = BertModel.from_pretrained(os.path.join(CURR_DIR, "bert-base-multilingual-cased"))
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        feature_extractor = BertModel.from_pretrained("bert-base-multilingual-cased")
        tokenizer.save_pretrained(os.path.join(CURR_DIR, "bert-base-multilingual-cased"))
        feature_extractor.save_pretrained(os.path.join(CURR_DIR, "bert-base-multilingual-cased"))
    feature_extractor.to(device)
    feature_extractor.eval()
    print("Successfully initialize pretrained BERT!")

    def bert_tokenizer(text):
        with torch.no_grad():
            encoded_input = tokenizer(text, return_tensors='pt', padding=True)
            for key in encoded_input:
                encoded_input[key] = encoded_input[key][:, :300].to(device)
            encoded_input["input_ids"][encoded_input["attention_mask"][:, -1] == 1][:, -1] = 102
        return encoded_input

    return bert_tokenizer, feature_extractor
