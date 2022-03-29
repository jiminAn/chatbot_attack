import numpy as np
from tqdm import tqdm
import os.path

import torch
from torch.utils.data import dataloader
from dialogLM.DataLoader import AutoRegressiveDataset
from dialogLM.Kogpt2 import DialogKoGPT2

root_path='./dialogLM'
data_path = f"{root_path}/data/wellness.user_chatbot.16k.train"
checkpoint_path =f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kogpt_wellness_epoch5_batch2.pth"

assert not os.path.isfile(save_ckpt_path) # avoid overlapping original model
assert torch.cuda.is_available()
n_epoch = 5
batch_size = 2
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)
save_step = 100
learning_rate = 5e-5

dataset= AutoRegressiveDataset(data_path)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DialogKoGPT2()
model.to(device)


loss_fct = torch.nn.CrossEntropyLoss(ignore_index=3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses =[]
for epoch in range(n_epoch):
    count = 0
    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = torch.stack(data)
            data = data.transpose(1, 0)
            data= data.to(ctx)

            outputs = model(data, labels=data)
            _, logits = outputs[:2]

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = data[..., 1:].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # if count % 10 == 0:
            #     print('epoch no.{} train no.{}  loss = {}'.format(epoch, count + 1, loss))
            if (count > 0 and count % save_step == 0) or (len(data) < batch_size):
                torch.save({
                    'epoch': epoch,
                    'train_no': count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_ckpt_path)
            count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")
