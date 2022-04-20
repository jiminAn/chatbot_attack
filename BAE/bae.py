import sys
import torch
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, util
from dialogLM.Kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer
import time
assert torch.cuda.is_available()
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

ans_data_path = "./dialogLM/data/wellness.user_chatbot.16k.test.ans"
answers = []
with open(ans_data_path, 'rt', encoding="UTF8") as f:
    for line in f.readlines():
        answers.append(line.strip())

save_ckpt_path = './dialogLM/checkpoint/kogpt_wellness_epoch5_batch2.pth'
kogpt2 = DialogKoGPT2()
checkpoint = torch.load(save_ckpt_path, map_location=device)
kogpt2.load_state_dict(checkpoint['model_state_dict'])
kogpt2.eval()
kogpt2_tokenizer = get_kogpt2_tokenizer()
output_size = 200

embedder = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForMaskedLM.from_pretrained("klue/bert-base")
import heapq
from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def get_kogpt2_ans(q):
    t_idx = kogpt2_tokenizer.encode(q.strip())
    result = kogpt2.generate(
        input_ids=torch.tensor([kogpt2_tokenizer.bos_token_id, ] + t_idx + [kogpt2_tokenizer.eos_token_id]).unsqueeze(0))
    ans = kogpt2_tokenizer.decode(result[0].tolist()[len(t_idx) + 1:], skip_special_tokens=True)
    return ans

def filtering(t, p_t):
    if t == p_t:
        return False
    if '[' in p_t:
        return False
    if '#' in p_t:
        return False
    return True

attack_sent_file = sys.argv[1]

start = time.time()  # 시작 시간 저장

with open(attack_sent_file, 'rt', encoding="UTF8") as f:
    for line in f.readlines()[:1]:
        idx, q = line.split('\t')
        sent = q.strip()
        idx = int(idx)
        tokens = sent.split()
        orig_q_emb = embedder.encode(sent)
        tokens_importance = []
        for i in range(len(tokens)):
            comp_token = tokens[0:i] + tokens[i + 1:]
            comp_q = (' ').join(comp_token)
            comp_q_emb = embedder.encode(comp_q)
            cos_similarity = cos_sim(orig_q_emb, comp_q_emb)
            TI = 1 - cos_similarity
            heapq.heappush(tokens_importance, (-TI, tokens[i]))
            #print(comp_q, cos_similarity, tokens[i], TI, sep='\t')
        print(sent)
        attack_success = False
        while tokens_importance and not attack_success:
            TI, token = heapq.heappop(tokens_importance) # 받았어
            #print(abs(TI), token, sep ='\t')
            S_m = sent.replace(token, '[MASK]')
            print(S_m)
            predict_mask_tokens_5_infos = pip(S_m)
            for i, info in enumerate(predict_mask_tokens_5_infos): # 후보 T
                pred_token = info['token_str']
                if filtering(token, pred_token):
                    #S_adv = S_m.replace('[MASK]',pred_token)
                    S_adv = info['sequence']
                    #print(S_adv)
                    ans_adv = get_kogpt2_ans(S_adv)
                    ans = answers[idx]
                    len_ans = len(ans)
                    ans_adv = ans_adv[:len_ans]
                    ans_emb = embedder.encode(ans)
                    ans_adv_emb = embedder.encode(ans_adv)
                    cos_similarity = cos_sim(ans_emb, ans_adv_emb)
                    print(q, ans, ans_adv, cos_similarity, sep ='\t')
                    if cos_similarity <= 0.4:
                        attack_success = True
                        break





print(time.time() - start)




