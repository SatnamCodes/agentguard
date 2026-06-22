from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf",data_dir="red-team-attempts")
print(dataset)
print(dataset['train'][0])