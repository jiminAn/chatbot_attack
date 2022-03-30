import torch
import torch.nn as nn
import random
from transformers import (
    ElectraConfig,
    ElectraTokenizer,

)

from koelectra import koElectraForSequenceClassification, koelectra_input
class DialogElectra:
    def __init__(self):
        self.checkpoint_path = "./checkpoint/"
        self.save_ckpt_path = self.checkpoint_path +  "model9.pth"
        model_path = "monologg/koelectra-base-discriminator"

        #self.category, self.answer = load_wellness_answer()

        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(ctx)

        checkpoint = torch.load(self.save_ckpt_path, map_location=self.device)

        # Electra Tokenizer
        self.tokenizer = ElectraTokenizer.from_pretrained(model_path)

        electra_config = ElectraConfig.from_pretrained(model_path)
        self.model = koElectraForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config=electra_config,
            num_labels=359)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentence):
        data = koelectra_input(self.tokenizer, sentence, self.device, 512)
        output = self.model(**data)

        logit = output
        softmax_logit = nn.Softmax(logit).dim
        softmax_logit = softmax_logit[0].squeeze()

        max_index = torch.argmax(softmax_logit).item()
        max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

        #answer_list = self.answer[self.category[str(max_index)]]
        #answer_len = len(answer_list) - 1
        #answer_index = random.randint(0, answer_len)

        # return answer_list
        # return self.category[str(max_index)]
        return max_index