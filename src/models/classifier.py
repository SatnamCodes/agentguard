import torch
import torch.nn as nn
class TraceClassifier(nn.Module):
    def __init__ (self, vocab_size = 30522, embed_dim = 128, num_classes = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self , input_ids):
        x = self.embedding(input_ids).mean(dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
if __name__ == "__main__":
    model  = TraceClassifier()
    dummy_input  = torch.randint(0 , 30522,(8,50))
    output  = model(dummy_input)
    print("Model ready. Output shape: ", output.shape)
