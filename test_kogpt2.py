import torch
from dialogLM.Kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer
assert torch.cuda.is_available()
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

save_ckpt_path = './dialogLM/checkpoint/kogpt_wellness_epoch5_batch2.pth'
model = DialogKoGPT2()
checkpoint = torch.load(save_ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = get_kogpt2_tokenizer()
output_size = 200

test_data_path = './dialogLM/data/wellness.user_chatbot.16k.test.ans'
questions = []
with open(test_data_path, 'rt', encoding='UTF8') as f:
    for line in f.readlines():
        q, a = line.split('\t')
        questions.append(q.strip())

t_idxs = [tokenizer.encode(q) for q in questions]
input_idxs = [torch.tensor([tokenizer.bos_token_id,]+t_idx+[tokenizer.eos_token_id]).unsqueeze(0) for t_idx in t_idxs]

for input_idx, t_idx in zip(input_idxs,t_idxs):
    result = model.generate(input_ids=input_idx)
    answer = tokenizer.decode(result[0].tolist()[len(t_idx)+1:],skip_special_tokens=True)
    print(answer)