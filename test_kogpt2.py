import torch
import time
from dialogLM.Kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer
assert torch.cuda.is_available()
start = time.time()  # 시작 시간 저장
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

save_ckpt_path = './dialogLM/checkpoint/kogpt_wellness_epoch5_batch2.pth'
model = DialogKoGPT2()
checkpoint = torch.load(save_ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = get_kogpt2_tokenizer()
output_size = 200

test_data_path = './dialogLM/data/wellness.user_chatbot.16k.test.q'

with open(test_data_path, 'rt', encoding='UTF8') as f:
    for q in f.readlines():
        t_idx = tokenizer.encode(q.strip())
        result = model.generate(input_ids=torch.tensor([tokenizer.bos_token_id,]+t_idx+[tokenizer.eos_token_id]).unsqueeze(0))
        print(tokenizer.decode(result[0].tolist()[len(t_idx)+1:],skip_special_tokens=True))

print(time.time()-start)