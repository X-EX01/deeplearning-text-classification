import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Import RNN models and tokenizer from rnn.py
from rnn import BiLSTM, BiGRU, tokenize as rnn_tokenize

CLASS_NAMES = [
    "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
    "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal",
    "Plant", "Album", "Film", "Written Work"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Load Models
# ==========================================
print("Loading saved models...")

# Paths updated to match train.py/test.py naming conventions
MODELS_DIR = "models" 

# Transformers
distilbert_tokenizer = AutoTokenizer.from_pretrained(f"{MODELS_DIR}/distilbert")
distilbert_model = AutoModelForSequenceClassification.from_pretrained(f"{MODELS_DIR}/distilbert").to(device).eval()

roberta_tokenizer = AutoTokenizer.from_pretrained(f"{MODELS_DIR}/roberta")
roberta_model = AutoModelForSequenceClassification.from_pretrained(f"{MODELS_DIR}/roberta").to(device).eval()

# RNN Shared Data
rnn_vocab = torch.load(f"{MODELS_DIR}/rnn_base/rnn_vocab.pth", weights_only=False)
vocab_size = len(rnn_vocab)
dummy_matrix = torch.zeros((vocab_size, 100))

# BiLSTM
bilstm_model = BiLSTM(100, 128, 14, dummy_matrix).to(device)
bilstm_model.load_state_dict(torch.load(f"{MODELS_DIR}/bilstm/bilstm_weights.pth", weights_only=True, map_location=device))
bilstm_model.eval()

# BiGRU
bigru_model = BiGRU(100, 128, 14, dummy_matrix).to(device)
bigru_model.load_state_dict(torch.load(f"{MODELS_DIR}/bigru/bigru_weights.pth", weights_only=True, map_location=device))
bigru_model.eval()

# ==========================================
# 2. Helper Functions
# ==========================================
def format_tensor(t):
    if isinstance(t, list): return str(t)
    t = t.detach().cpu().numpy()
    shape_str = str(list(t.shape))
    content_snippet = str(np.round(t.flatten()[:5], 4).tolist()) + "..."
    return shape_str, content_snippet

def make_block(title, shape, content, color1, color2, is_last=False):
    hover_text = f"TENSOR SHAPE: {shape}&#10;SAMPLE DATA: {content}"
    html = f'''
    <div style="background: linear-gradient(135deg, {color1} 0%, {color2} 100%); 
                color: white; padding: 12px; border-radius: 8px; margin: 5px auto; 
                width: 95%; text-align: center; font-weight: bold; cursor: crosshair; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.2); transition: transform 0.2s; font-size: 0.85em;" 
         onmouseover="this.style.transform='scale(1.03)'" 
         onmouseout="this.style.transform='scale(1)'"
         title="{hover_text}">
        {title}
    </div>
    '''
    if not is_last:
        html += '<div style="text-align: center; color: gray; font-size: 18px; margin: -5px 0;">⬇</div>'
    return html

# ==========================================
# 3. Processing Logic
# ==========================================
def get_transformer_html(text, model, tokenizer, colors):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits[0], dim=0)
    
    pred = CLASS_NAMES[torch.argmax(probs).item()]
    
    html = ""
    s, c = format_tensor(inputs['input_ids'])
    html += make_block("1. Tokenizer (Subword)", s, c, colors[0], colors[1])
    
    s, c = format_tensor(hidden_states[0])
    html += make_block("2. Multi-Head Embeds", s, c, colors[0], colors[1])
    
    # Show ALL transformer blocks
    num_layers = len(hidden_states) - 1
    for i in range(1, num_layers + 1):
        s, c = format_tensor(hidden_states[i])
        html += make_block(f"3.{i} Transformer Block", s, c, colors[2], colors[3])
        
    s, c = format_tensor(logits)
    html += make_block(f"4. Classification Head: {pred}", s, c, "#11998e", "#38ef7d", is_last=True)
    return html

def get_rnn_html(text, model, is_lstm, colors):
    tokens = rnn_tokenize(text)
    token_ids = [rnn_vocab.get(t, 1) for t in tokens][:128]
    if not token_ids: token_ids = [1]
    input_tensor = torch.tensor([token_ids]).to(device)
    
    with torch.no_grad():
        embeds = model.embedding(input_tensor)
        # Handle BiLSTM vs BiGRU output tuple differences
        if is_lstm:
            rnn_out, (hn, cn) = model.rnn(embeds)
        else:
            rnn_out, hn = model.rnn(embeds)
            
        final_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        logits = model.fc(final_hidden)
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        
    pred = CLASS_NAMES[torch.argmax(probs).item()]
    
    html = ""
    s, c = format_tensor(input_tensor)
    html += make_block("1. Word Tokenizer", s, c, colors[0], colors[1])
    
    s, c = format_tensor(embeds)
    html += make_block("2. Static GloVe Embeds", s, c, colors[0], colors[1])
    
    layer_name = "Bi-LSTM Layer" if is_lstm else "Bi-GRU Layer"
    s, c = format_tensor(rnn_out)
    html += make_block(f"3. {layer_name}", s, c, colors[2], colors[3])
    
    s, c = format_tensor(logits)
    html += make_block(f"4. Classification Head: {pred}", s, c, "#11998e", "#38ef7d", is_last=True)
    return html

def generate_all_flowcharts(text):
    if not text.strip():
        return ["Please enter text"] * 4

    # Themes (StartColor, EndColor, LayerColor1, LayerColor2)
    distil_theme = ["#1e3c72", "#2a5298", "#2b5876", "#4e4376"]
    rob_theme = ["#4b6cb7", "#182848", "#00c6ff", "#0072ff"]
    lstm_theme = ["#cb2d3e", "#ef473a", "#d53369", "#cbad6d"]
    gru_theme = ["#f85032", "#e73827", "#f7971e", "#ffd200"]

    d_html = get_transformer_html(text, distilbert_model, distilbert_tokenizer, distil_theme)
    r_html = get_transformer_html(text, roberta_model, roberta_tokenizer, rob_theme)
    l_html = get_rnn_html(text, bilstm_model, True, lstm_theme)
    g_html = get_rnn_html(text, bigru_model, False, gru_theme)

    return d_html, r_html, l_html, g_html

# ==========================================
# 4. Gradio UI
# ==========================================
with gr.Blocks(title="Deep Learning Architecture X-Ray") as demo:
    gr.Markdown("# Multimodal Architectural Comparison")
    gr.Markdown("Compare how different backbones process the same text. **Hover over blocks** to see tensor shapes.")
    
    text_input = gr.Textbox(lines=2, placeholder="Type something...", label="Input Text")
    btn = gr.Button("Analyze All Models", variant="primary")
    
    # ----------------------------------------------------
    # ALL FOUR MODELS ON A SINGLE ROW
    # ----------------------------------------------------
    with gr.Row():
        with gr.Column(min_width=150):
            gr.Markdown("### DistilBERT")
            distil_out = gr.HTML()
            
        with gr.Column(min_width=150):
            gr.Markdown("### RoBERTa")
            rob_out = gr.HTML()

        with gr.Column(min_width=150):
            gr.Markdown("### Bi-LSTM")
            lstm_out = gr.HTML()
            
        with gr.Column(min_width=150):
            gr.Markdown("### Bi-GRU")
            gru_out = gr.HTML()
            
    btn.click(
        fn=generate_all_flowcharts, 
        inputs=text_input, 
        outputs=[distil_out, rob_out, lstm_out, gru_out]
    )

if __name__ == "__main__":
    demo.launch()