from datasets import load_dataset
import json
import os

os.makedirs("src/data", exist_ok=True)

# Correct load for red-team data
dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
train_data = dataset['train']

traces = []
for example in train_data.select(range(20000)):
    transcript = example['transcript']
    score = example.get('min_harmlessness_score_transcript', 0)
    label = 1 if score < 0 else 0
    traces.append({
        "text": transcript[:2000],
        "label": label,
        "score": float(score)
    })

with open("src/data/processed_traces.json", "w") as f:
    json.dump(traces, f, indent=2)

print(f"Saved {len(traces)} traces")