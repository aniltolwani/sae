import torch
from transformers import AutoTokenizer, AutoModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Load model and tokenizer
model_name = "bert-base-uncased"  # Or other model of choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

def get_attention_and_similarity(text, model, tokenizer):
   # Tokenize and get model outputs
   inputs = tokenizer(text, return_tensors="pt")
   outputs = model(**inputs)
   
   # Get embeddings
   embeddings = outputs.last_hidden_state.squeeze(0)
   
   # Calculate cosine similarities between tokens
   similarities = torch.nn.functional.cosine_similarity(
       embeddings.unsqueeze(1), 
       embeddings.unsqueeze(0), 
       dim=2
   )
   
   # Average attention across all heads and layers
   attentions = outputs.attentions  # Tuple of attention tensors
   avg_attention = torch.mean(torch.stack([
       layer_attention.squeeze(0) for layer_attention in attentions
   ]), dim=(0,1))  # Average across layers and heads
   
   return avg_attention, similarities, tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

def plot_matrices(text, attention, similarity, tokens):
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
   
   # Plot attention heatmap
   sns.heatmap(attention.detach().numpy(), ax=ax1, xticklabels=tokens, yticklabels=tokens)
   ax1.set_title("Attention Pattern")
   
   # Plot similarity heatmap
   sns.heatmap(similarity.detach().numpy(), ax=ax2, xticklabels=tokens, yticklabels=tokens)
   ax2.set_title("Similarity Pattern")
   
   plt.tight_layout()
   return fig

def analyze_patterns(attention, similarity, tokens):
   # Get average attention and similarity for each token
   avg_attention = attention.mean(dim=1)
   avg_similarity = similarity.mean(dim=1)
   
   # Define thresholds (you might want to tune these)
   att_threshold = avg_attention.mean()
   sim_threshold = avg_similarity.mean()
   
   # Categorize tokens
   categories = {
       "high_att_low_sim": [],
       "low_att_high_sim": [],
       "high_att_high_sim": [],
       "low_att_low_sim": []
   }
   
   for i, token in enumerate(tokens):
       if avg_attention[i] > att_threshold:
           if avg_similarity[i] > sim_threshold:
               categories["high_att_high_sim"].append(token)
           else:
               categories["high_att_low_sim"].append(token)
       else:
           if avg_similarity[i] > sim_threshold:
               categories["low_att_high_sim"].append(token)
           else:
               categories["low_att_low_sim"].append(token)
               
   return categories

def save_analysis_results(text, patterns, fig, output_dir="debug_output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"analysis_{timestamp}"
    
    # Save patterns to text file
    with open(os.path.join(output_dir, f"{base_filename}.txt"), "w") as f:
        f.write(f"Analysis for: {text}\n\n")
        for category, tokens in patterns.items():
            f.write(f"{category}:\n")
            f.write(f"{tokens}\n\n")
    
    # Save figure
    fig.savefig(os.path.join(output_dir, f"{base_filename}.png"))
    plt.close(fig)

# Test examples
examples = [
   "Berlin has a population of 3.8M people.",
   "The quick red fox jumped over the lazy brown dog.",
]

for text in examples:
   print(f"\nAnalyzing: {text}")
   attention, similarity, tokens = get_attention_and_similarity(text, model, tokenizer)
   
   # Plot matrices
   fig = plot_matrices(text, attention, similarity, tokens)
   
   # Analyze patterns
   patterns = analyze_patterns(attention, similarity, tokens)
   
   # Save results
   save_analysis_results(text, patterns, fig)
   
   # Optional: still print to console if desired
   for category, tokens in patterns.items():
       print(f"\n{category}:")
       print(tokens)