import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,GPT2Model
from torch.utils.data import Dataset, DataLoader

import pandas as pd

# from transformers import cached_download

# # 清除缓存
# cached_download.clear_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载微调前的GPT-2模型和tokenizer
# 定义微调参数
# data_path = news  # 数据集文件路径
batch_size = 4
max_length = 512  # 根据数据集和模型的最大输入长度来设置
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained(model_name)
df=pd.read_csv('bbc_news.csv')
column_data = df['description'] # 读取description列的数据
news = list(column_data) # 将description列的数据转换成列表
news=news[:2000]
# 加载BBC News数据集并进行预处理
class NewsDataset(Dataset):
    def __init__(self, news, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attention_masks = []
        
        news_data=news

        for news in news_data:
            encoded_news = tokenizer.encode_plus(
                news,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
#             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.input_ids.append(encoded_news['input_ids'])
            self.attention_masks.append(encoded_news['attention_mask'])
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index].squeeze(),
            'attention_mask': self.attention_masks[index].squeeze()
        }



# 加载数据集
dataset = NewsDataset(news, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 将模型移动到设备上
model.to(device)

# 定义微调参数
num_epochs = 5
learning_rate = 1e-4

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# 开始微调
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

# 保存微调后的模型
output_model_dir = './fine_tuned_model/'
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)

# 使用微调后的模型生成新闻
def generate_news(model, tokenizer, prompt_text, max_length):
    model.eval()
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            early_stopping=True
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# 加载微调后的模型和tokenizer
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_model_dir)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_model_dir)

# 设置生成新闻的参数
prompt = "Today's news: "
max_length = 200

# 生成新闻
generated_news = generate_news(fine_tuned_model, fine_tuned_tokenizer, prompt, max_length)
print(generated_news)