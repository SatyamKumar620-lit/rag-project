from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import pandas as pd
import torch
import os

# ✅ Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ✅ Load dataset and limit size
dataset = load_dataset("Abirate/english_quotes", split="train")
df = dataset.to_pandas()

# ✅ Drop missing and sample only what's available (max 3000 rows)
df = df.dropna(subset=['quote', 'author'])
df = df.sample(n=min(3000, len(df)), random_state=42)
df['text'] = df['quote'] + " - " + df['author']

# ✅ Save cleaned data for later use
df.to_csv("cleaned_quotes.csv", index=False)

# ✅ Convert to InputExample for sentence-transformers
train_examples = [InputExample(texts=[row['text'], row['text']]) for _, row in df.iterrows()]

# ✅ Define model architecture
word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# ✅ Load model on GPU
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

# ✅ Training config
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.MultipleNegativesRankingLoss(model)

# ✅ Fine-tune (1 epoch for speed/memory)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)

# ✅ Save model
model.save("finetuned_model")
print("✅ Model training complete and saved to 'finetuned_model/'")
