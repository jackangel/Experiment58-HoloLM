import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests
import os
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# --- Hyperparameters ---
# Increased vocab size for BPE (vs ~65 for chars)
VOCAB_SIZE = 4096   
EMBED_DIM = 512
MATRIX_DIM = 64
NUM_LAYERS = 2
SEQ_LEN = 128
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
STEPS = 50000        # Reduced steps slightly as BPE learns faster per step
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'holo_checkpoint.pth'
TOKENIZER_FILE = 'holo_tokenizer.json'

# --- 1. JIT Compiled Mathematical Core ---
@torch.jit.script
def holo_scan(x, memory, 
              w_k, b_k, w_q, b_q, w_v, b_v, w_out, b_out, 
              w_gw, b_gw, w_gf, b_gf):
    """
    Scans a sequence using the Holographic Memory update rule.
    """
    outputs: list[torch.Tensor] = [] # Type annotation for JIT safety
    
    # Pre-compute projections
    k_all = F.normalize(F.linear(x, w_k, b_k), p=2.0, dim=-1)
    q_all = F.normalize(F.linear(x, w_q, b_q), p=2.0, dim=-1)
    v_all = torch.tanh(F.linear(x, w_v, b_v))
    
    gw_all = torch.sigmoid(F.linear(x, w_gw, b_gw)).unsqueeze(-1)
    gf_all = torch.sigmoid(F.linear(x, w_gf, b_gf)).unsqueeze(-1)
    
    # Loop over time
    for t in range(x.size(1)):
        k = k_all[:, t]
        q = q_all[:, t]
        v = v_all[:, t]
        beta = gw_all[:, t]
        decay = gf_all[:, t]
        
        # READ
        readout = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        
        # WRITE
        association = torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
        memory = (decay * memory) + (beta * association)
        
        outputs.append(readout)
        
    # Stack and final project
    stacked = torch.stack(outputs, dim=1)
    final_out = F.linear(stacked, w_out, b_out)
    
    return final_out, memory

# --- 2. The Module Wrapper ---
class FastHoloBlock(nn.Module):
    def __init__(self, embed_dim, matrix_dim):
        super().__init__()
        self.matrix_dim = matrix_dim
        
        self.proj_k = nn.Linear(embed_dim, matrix_dim)
        self.proj_q = nn.Linear(embed_dim, matrix_dim)
        self.proj_v = nn.Linear(embed_dim, matrix_dim)
        self.proj_out = nn.Linear(matrix_dim, embed_dim)
        
        self.gate_write = nn.Linear(embed_dim, 1)
        self.gate_forget = nn.Linear(embed_dim, 1)
        
        nn.init.constant_(self.gate_write.bias, -2.0)
        nn.init.constant_(self.gate_forget.bias, 2.0)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x, memory=None):
        B = x.shape[0]
        if memory is None:
            memory = torch.zeros(B, self.matrix_dim, self.matrix_dim, device=x.device)
            
        x_norm = self.ln1(x)
        
        mixer_out, new_memory = holo_scan(
            x_norm, memory,
            self.proj_k.weight, self.proj_k.bias,
            self.proj_q.weight, self.proj_q.bias,
            self.proj_v.weight, self.proj_v.bias,
            self.proj_out.weight, self.proj_out.bias,
            self.gate_write.weight, self.gate_write.bias,
            self.gate_forget.weight, self.gate_forget.bias
        )
        
        x = x + mixer_out
        x = x + self.ffn(self.ln2(x))
        
        return x, new_memory

class HoloGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, matrix_dim, layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            FastHoloBlock(embed_dim, matrix_dim) for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, idx, states=None):
        x = self.token_emb(idx)
        
        new_states = []
        if states is None:
            states = [None] * len(self.layers)
            
        for i, layer in enumerate(self.layers):
            x, s = layer(x, states[i])
            new_states.append(s)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_states

# --- 3. Tokenizer & Data Logic ---
def get_tokenizer_and_data():
    # Download data if missing
    if not os.path.exists('input.txt'):
        print("Downloading input.txt...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        r = requests.get(url)
        with open('input.txt', 'w') as f: f.write(r.text)

    # Train or Load Tokenizer
    if os.path.exists(TOKENIZER_FILE):
        print(f"Loading existing tokenizer from {TOKENIZER_FILE}...")
        tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    else:
        print("Training new BPE tokenizer...")
        # Initialize BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        
        # Pre-tokenizer: Split by byte level but DO NOT add prefix space automatically
        # This ensures 'Hello' is 'Hello', and ' Hello' is ' Hello'.
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False) 
        
        # Decoder: To reverse the process
        tokenizer.decoder = ByteLevelDecoder()
        
        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE, 
            min_frequency=2, 
            special_tokens=["[UNK]", "[PAD]"]
        )
        
        tokenizer.train(["input.txt"], trainer)
        tokenizer.save(TOKENIZER_FILE)
        print("Tokenizer trained and saved.")

    # Prepare Data Tensor
    with open('input.txt', 'r') as f: text = f.read()
    
    # Encode the entire text
    print("Encoding dataset...")
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    
    return data, tokenizer

def sample_top_p_top_k(logits, temperature=1.0, top_k=0, top_p=0.0):
    """
    logits: [batch_size, vocab_size] (last step)
    temperature: >0.0 (scales probability distribution)
    top_k: >0 (keep only top k tokens)
    top_p: >0.0 (nucleus sampling, keep top tokens summing to p)
    """
    # 1. Temperature scaling
    logits = logits / temperature
    
    # 2. Top-K Filtering
    if top_k > 0:
        # Keep only top k, mask others with -inf
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v[:, -1].unsqueeze(1)
        logits[logits < pivot] = -float('Inf')

    # 3. Top-P (Nucleus) Filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        
    return logits

def main():
    print(f"Initializing Fast HoloGPT (BPE Edition) on {DEVICE}...")
    
    # Load Data & Tokenizer
    data, tokenizer = get_tokenizer_and_data()
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab Size: {actual_vocab_size}")

    model = HoloGPT(actual_vocab_size, EMBED_DIM, MATRIX_DIM, NUM_LAYERS).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nCheckpoint '{CHECKPOINT_PATH}' found! Loading...")
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            print("Loaded successfully.")
        except Exception as e:
            print(f"Checkpoint mismatch ({e}). Starting fresh.")
    else:
        # --- Start Training ---
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(range(STEPS), desc="Training")
        
        for step in pbar:
            ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
            xb = torch.stack([data[i:i+SEQ_LEN] for i in ix]).to(DEVICE)
            yb = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix]).to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(xb)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), yb.view(B*T))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 50 == 0:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
        print(f"\nSaving checkpoint to {CHECKPOINT_PATH}...")
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("TRAINING COMPLETE.")

# --- Chat Interface ---
    print("\nStarting Chat Mode.")
    print("Settings: Temp=0.8, Top-K=50, Top-P=0.9")
    print("(Type 'exit' to quit)")
    
    model.eval()
    
    # Generation Hyperparameters
    TEMPERATURE = 0.8
    TOP_K = 50
    TOP_P = 0.9

    while True:
        try:
            prompt = input("\nYou: ")
        except EOFError:
            break

        if not prompt or prompt.strip() == "":
            continue
            
        if prompt.lower() == "exit": break
        
        # 1. Tokenize Input
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        
        # Warmup memory
        _, states = model(idx)
        
        print(f"Holo: {prompt}", end="", flush=True)
        curr = idx[:, -1:]
        
        # Generate
        with torch.no_grad():
            for _ in range(200): # Generation length
                logits, states = model(curr, states)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :]
                
                # Apply Top-K / Top-P / Temperature
                filtered_logits = sample_top_p_top_k(
                    next_token_logits, 
                    temperature=TEMPERATURE, 
                    top_k=TOP_K, 
                    top_p=TOP_P
                )
                
                # Convert to probabilities
                probs = F.softmax(filtered_logits, dim=-1)
                
                # Sample
                next_token_id = torch.multinomial(probs, 1)
                
                # Decode Token
                decoded_char = tokenizer.decode([next_token_id.item()])
                
                print(decoded_char, end="", flush=True)
                curr = next_token_id
        print()

if __name__ == "__main__":
    main()