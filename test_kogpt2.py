import torch
import time
import sys
from dialogLM.Kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer
assert torch.cuda.is_available()
from transformers import AutoModel, AutoTokenizer
model_name = "taeminlee/kogpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
from numpy.linalg import norm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np
from numpy import dot
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

start = time.time()  # 시작 시간 저장
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

save_ckpt_path = './dialogLM/checkpoint/kogpt_wellness_epoch5_batch2.pth'
model = DialogKoGPT2()
checkpoint = torch.load(save_ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
embedder = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
#tokenizer = get_kogpt2_tokenizer()
output_size = 200

q_ans_data_path = './dialogLM/data/wellness.user_chatbot.16k.test'
q_answers = []
with open(q_ans_data_path, 'rt', encoding='UTF8') as f:
    for line in f.readlines():
        q, a = line.split('\t')
        q_answers.append((q.strip(), a.strip()))

test_data_path = sys.argv[1]
s_adv_candis = defaultdict(list)
with open(test_data_path, 'rt', encoding='UTF16') as f:
    for line in f.readlines():
        idx, q = line.split('\t')
        s_adv_candis[int(idx)].append(q.strip())

attack_success = 0
for idx in s_adv_candis.keys():
    ans = q_answers[idx][1]
    orig_q = q_answers[idx][0]
    len_ans = len(ans)
    for q in s_adv_candis[idx]:
        t_idx = tokenizer.encode(q.strip())
        result = model.generate(input_ids=torch.tensor([tokenizer.bos_token_id,]+t_idx+[tokenizer.eos_token_id]).unsqueeze(0))
        s_adv_ans = tokenizer.decode(result[0].tolist()[len(t_idx)+1:],skip_special_tokens=True)[:len_ans]
        ans_emb = embedder.encode(ans)
        g_ans_emb = embedder.encode(s_adv_ans)
        cos_simirality = cos_sim(ans_emb, g_ans_emb)
        print(idx, orig_q, q, ans, s_adv_ans, cos_simirality, sep='\t')
        if cos_simirality < 0.4:
            attack_success += 1
            break

print(attack_success)
print(time.time()-start)