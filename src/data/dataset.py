from datasets import load_dataset
import json
dataset = load_dataset("Anthropic/hh-rlhf",data_dir="red-team-attempts")
train_data = dataset['train']

#Process
traces = []
for example in train_data.select(range(10000)):
    transcript = example['transcript']
    score = example.get('min_harmlessness_score_transcript' , 0)
    label = 1 if score<0 else 0
    traces.append({"text":transcript,"label": label , "score": float(score)})

#Save
with open("src/data/processed_traces.json" , "w") as f:
    json.dump(traces,f,indent = 2)
print(f"Saved{len(traces)} traces.")