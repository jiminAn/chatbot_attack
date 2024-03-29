import sys
import torch
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, util
import time
import re
from konlpy.tag import Mecab
assert torch.cuda.is_available()
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)
embedder = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
mecab = Mecab(dicpath='C:\mecab\mecab-ko-dic')
special = re.compile(r'[^ A-Za-z0-9가-힣+]')

from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForMaskedLM.from_pretrained("klue/bert-base")
import heapq
from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)


def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))


def filtering(t, p_t):
    if t == p_t:
        return False
    if mecab.pos(t)[0][1] != mecab.pos(p_t)[0][1]:
        return False
    if '[' in p_t:
        return False
    if '#' in p_t:
        return False
    return True

attack_sent_file = sys.argv[1]

start = time.time()  # 시작 시간 저장

with open(attack_sent_file, 'rt', encoding="UTF8") as f:
    for line in f.readlines():
        idx, q = line.split('\t')
        sent = q.strip()
        idx = int(idx)
        #tokens = sent.split()
        pre_sent = special.sub('',sent)
        tokens = mecab.morphs(pre_sent)
        orig_q_emb = embedder.encode(pre_sent)
        tokens_importance = []
        for i in range(len(tokens)):
            comp_token = tokens[0:i] + tokens[i + 1:]
            comp_q = (' ').join(comp_token)
            comp_q_emb = embedder.encode(comp_q)
            cos_similarity = cos_sim(orig_q_emb, comp_q_emb)
            TI = 1 - cos_similarity
            heapq.heappush(tokens_importance, (-TI, tokens[i]))
            #print(comp_q, cos_similarity, tokens[i], TI, sep='\t')
        #print(sent)
        while tokens_importance:
            TI, token = heapq.heappop(tokens_importance) # 받았어
            #print(abs(TI), token, sep ='\t')
            S_m = sent.replace(token, '[MASK]')
            if '[MASK]' not in S_m:
                continue
            predict_mask_tokens_5_infos = pip(S_m)
            #print(S_m)
            for i, info in enumerate(predict_mask_tokens_5_infos): # 후보 T
                if 'token_str' in info:
                    pred_token = info['token_str']
                    if filtering(token, pred_token):
                        #S_adv = S_m.replace('[MASK]',pred_token)
                        S_adv = info['sequence']
                        emb_s = embedder.encode(sent)
                        emb_s_adv = embedder.encode(S_adv)
                        cos = cos_sim(emb_s, emb_s_adv)
                        epsilon = 0.75
                        #print(S_adv, cos, sep='\t')
                        if cos > epsilon:
                            print(idx, S_adv, sep='\t')

print(time.time() - start)




