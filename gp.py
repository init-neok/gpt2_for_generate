from transformers import GPT2Tokenizer
import torch
from transformers import AdamW
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
def preprocess(text):
    encoding = tokenizer.encode_plus(text,max_length=512 ,truncation=True, padding='max_length', return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask

#train函数的定义


def train(model, optimizer, epochs, training_loader):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        loss = 0
        for i, batch in enumerate(training_loader):
            inputs, targets = batch
#             print(inputs.shape,targets.shape)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if (i+1) % 20== 0:
                print('Epoch {:2d} | Batch {:2d}/{:2d} | Loss {:.4f}'.format(epoch+1, i+1, len(training_loader), loss))

        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pt')

#定义数据类
class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, attention_mask = preprocess(self.data.iloc[index]['description'])
        input_ids = torch.tensor(input_ids).squeeze()
        attention_mask = torch.tensor(attention_mask).squeeze()
        return input_ids, attention_mask

dataset = pd.read_csv('bbc_news.csv')[['description']]

training_set = NewsDataset(dataset[:2000])
training_loader = DataLoader(training_set, batch_size=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=GPT2LMHeadModel.from_pretrained('gpt2').to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
train(model, optimizer, 5, training_loader)


#测试脚本
text_input=torch.tensor(tokenizer.encode("Australian politicians are", add_special_tokens=True))
text_input=text_input.to(device)
generated = model.generate(
    input_ids = text_input.unsqueeze(0),
    max_length = 50,
    temperature = 1.0,
    top_k = 0,
    top_p = 0.9,
    do_sample=True,
    num_return_sequences=1
)
print(text_input)
generated_title = tokenizer.decode(generated[0], skip_special_tokens=True)
print(generated_title)
