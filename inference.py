import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd
import time
import faiss
import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Функции преобразования для bi-енкодера
def bi_mean_pool(token_embeds: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool


def bi_encode(input_texts, tokenizer: AutoTokenizer, model: AutoModel, device: str = "cpu"
) -> torch.tensor:

    model.eval()
    tokenized_texts = tokenizer(input_texts, max_length=128,
                                padding='max_length', truncation=True, return_tensors="pt")
    token_embeds = model(tokenized_texts["input_ids"].to(device),
                         tokenized_texts["attention_mask"].to(device)).last_hidden_state
    pooled_embeds = bi_mean_pool(token_embeds, tokenized_texts["attention_mask"].to(device))
    return pooled_embeds

# Определяем BI-енкодер
class BiEncoder(torch.nn.Module):
    def __init__(self, model_name, max_length: int = 128):
        super().__init__()
        self.max_length = max_length
        self.bert_model = AutoModel.from_pretrained(model_name)
        # self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size * 3, 3)

    def forward(self, data: datasets.arrow_dataset.Dataset) -> torch.tensor:
        premise_input_ids = data["premise_input_ids"].to(device)
        premise_attention_mask = data["premise_attention_mask"].to(device)
        hypothesis_input_ids = data["hypothesis_input_ids"].to(device)
        hypothesis_attention_mask = data["hypothesis_attention_mask"].to(device)

        out_premise = self.bert_model(premise_input_ids, premise_attention_mask)
        out_hypothesis = self.bert_model(hypothesis_input_ids, hypothesis_attention_mask)
        premise_embeds = out_premise.last_hidden_state
        hypothesis_embeds = out_hypothesis.last_hidden_state

        pooled_premise_embeds = mean_pool(premise_embeds, premise_attention_mask)
        pooled_hypotheses_embeds = mean_pool(hypothesis_embeds, hypothesis_attention_mask)

        embeds =  torch.cat([pooled_premise_embeds, pooled_hypotheses_embeds,
                             torch.abs(pooled_premise_embeds - pooled_hypotheses_embeds)],
                            dim=-1)
        return self.linear(embeds)

# Определяем кросс-енкодер
MAX_LENGTH = 128
class CrossEncoderBert(torch.nn.Module):
    def __init__(self, max_length: int = MAX_LENGTH):
        super().__init__()
        self.max_length = max_length
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        # self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the CLS token's output
        return self.linear(pooled_output)
    
    
def get_top_k(query, corpus, top_k=5):
    """
      Выбор k кандидатов. Bi-Encoder
    """
    bi_pooled_embeds = torch.tensor(corpus['pooled_embeds'].apply(json.loads))

    bi_pooled_embeds_query = bi_encode(query, bi_tokenizer, bi_model.bert_model, device)
    bi_pooled_embeds_query = bi_pooled_embeds_query.cpu().detach().numpy() 

    similarities = cosine_similarity(bi_pooled_embeds_query, bi_pooled_embeds)

    sim_indexies = np.argsort(similarities)[0, ::-1]
    sim_indexies = sim_indexies[:top_k]
    return corpus.iloc[sim_indexies], similarities[0, sim_indexies]


def get_ranked_docs(
    query: str, candidates,
    ce_tokenizer: AutoTokenizer, 
    finetuned_ce: CrossEncoderBert 
) -> None:
    """
        Реранкер Cross-Encoder
    """
    corpus = candidates['line'].to_list() 
    # print(corpus)

    queries = [query] * len(corpus)
    tokenized_texts = ce_tokenizer(
        queries, corpus, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # Finetuned CrossEncoder model scoring
    with torch.no_grad():
        ce_scores = finetuned_ce(tokenized_texts['input_ids'], tokenized_texts['attention_mask']).squeeze(-1)
        ce_scores = torch.sigmoid(ce_scores)  # Apply sigmoid if needed

    # Process scores for finetuned model
    # print(f"Query - {query} [Finetuned Cross-Encoder]\n---")
    scores = ce_scores.cpu().numpy()
    scores_ix = np.argsort(scores)[::-1]
    # for ix in scores_ix:  # Limit to corpus size
    #     print(f"{scores[ix]: >.2f}\t{corpus[ix]}")
        
    return candidates.iloc[scores_ix[0]]['response'], scores[scores_ix[0]]

# Определяем модели
bi_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
bi_model = BiEncoder("sentence-transformers/all-MiniLM-L6-v2", max_length=128)
bi_model.load_state_dict(torch.load('models/BI_model_last/BI_model.pth', weights_only=True, map_location=torch.device('cpu')))
bi_model.to(device)
bi_model.eval()

ce_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
ce_model = CrossEncoderBert()
ce_model.load_state_dict(torch.load('models/CE_model_last/CE_model.pth', weights_only=True, map_location=torch.device('cpu')))
ce_model.to(device)
ce_model.eval()

# Загружаем данные 
house_answers = pd.read_csv('data/house_answers.csv')

# Добавляем индекс
index = faiss.IndexFlatIP((384)) # размерность pooled слоя у bi енкодера
index.add(torch.tensor(house_answers['pooled_embeds'].apply(json.loads)))


def get_answer(query: str):
    """
        Возвращает релевантный ответ (без ускорения)
    """
    start_time = time.time()
    candidates = get_top_k(query, house_answers) 
    ranked_candidate = get_ranked_docs(query, candidates[0], ce_tokenizer, ce_model)
    end_time = time.time() - start_time
    return *ranked_candidate, end_time


def get_answer_faiss(query: str):
    """
        Возвращает релевантный ответ (ускоряем с помощью faiss-индекса)
    """
    start_time = time.time()
    bi_pooled_embeds_query = bi_encode(query, bi_tokenizer, bi_model.bert_model, device)
    bi_pooled_embeds_query = bi_pooled_embeds_query.cpu().detach().numpy() 
    candidates = index.search(bi_pooled_embeds_query, k=20) # создаем индекс
    candidates = house_answers.iloc[candidates[1][0]]
    
    ranked_candidate = get_ranked_docs(query, candidates, ce_tokenizer, ce_model)
    end_time = time.time() - start_time
    return *ranked_candidate, end_time


if __name__ == "__main__":
    query = "You can't go in there."
    # print(get_answer(query))
    print(get_answer_faiss(query)) # Ответ - Who are you, and why are you wearing a tie?