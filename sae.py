import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset
import modal

app = modal.App("sae-anil")
image = modal.Image.debian_slim().pip_install([
    "einops",
    "transformers[torch]",
    "datasets",
    # "torch",
    "tqdm"
])

def get_mlp_activations(model, tokenizer, n_samples=1000):
    """Get post-GELU activations from first MLP layer using The Pile dataset"""
    # Load a subset of The Pile
    dataset = load_dataset(
        "EleutherAI/pile",
        streaming=True,
        split="train"
    )
    
    activations = []
    
    def hook_fn(m, i, o):
        activations.append(o.detach())

    # Register hook on GELU activation
    hook = model.model.layers[0].mlp.act.register_forward_hook(hook_fn)
    
    # Get activations from forward pass
    with torch.no_grad():
        for item in dataset.take(n_samples):
            # Skip very short texts
            if len(item['text'].strip()) < 10:
                continue
                
            # Tokenize and move to GPU
            tokens = tokenizer(
                item['text'],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to("cuda")
            
            # Forward pass
            model(**tokens)
            
            # Print progress
            if len(activations) % 100 == 0:
                print(f"Processed {len(activations)} samples")
                
            if len(activations) >= n_samples:
                break
    
    hook.remove()
    
    # Stack and reshape: [batch * sequence_length, d_mlp]
    acts = torch.cat(activations, dim=0)
    print(f"Collected {acts.shape[0]} activation vectors")
    return einops.rearrange(acts, 'b s d -> (b s) d')

class SparseAutoencoder(nn.Module):
    def __init__(self, d_input, d_hidden):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_input)
        
    def forward(self, x):
        # Get hidden activations
        h = F.relu(self.encoder(x))
        # Reconstruction
        x_reconstruct = self.decoder(h)
        # L2 reconstruction loss + L1 sparsity loss
        loss = F.mse_loss(x_reconstruct, x) + 1e-3 * h.abs().mean()
        return loss, h

@app.function(
    image=image,
    gpu="A100",
    secrets=[
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("huggingface-api-key")
    ],
    timeout=3600
)
def train():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    print("Getting activations...")
    activations = get_mlp_activations(model, tokenizer, n_samples=1000)
    
    # Create dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize autoencoder 
    d_mlp = model.config.intermediate_size
    autoencoder = SparseAutoencoder(d_mlp, d_mlp * 2).cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    print("Training...")
    n_epochs = 5
    for epoch in range(n_epochs):
        total_loss = 0
        total_sparsity = 0
        n_batches = 0
        
        for batch, in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            optimizer.zero_grad()
            loss, h = autoencoder(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_sparsity += (h > 0).float().mean().item()
            n_batches += 1
            
            # Print running statistics every 10 batches
            if n_batches % 10 == 0:
                avg_loss = total_loss / n_batches
                avg_sparsity = total_sparsity / n_batches
                print(f"Epoch {epoch+1}, Batch {n_batches}")
                print(f"Average Loss: {avg_loss:.3f}, Average Sparsity: {avg_sparsity:.3f}")

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'sae_checkpoint_epoch_{epoch+1}.pt')

@app.local_entrypoint()
def main():
    train.remote()

if __name__ == "__main__":
    main()