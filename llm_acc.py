# bleu score
import nltk
from nltk.translate.bleu_score import corpus_bleu
reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'a', 'test', 'but']
bleu_score = corpus_bleu([reference], [candidate], weights=(1,2))
print("BLEU Score:", bleu_score)
'''
    # ROUGE-1（Unigram ROUGE）：
    # ROUGE-1考虑的是单个词语的匹配情况。它计算生成文本和参考答案中相同的单个词语数量的比例，从而衡量了词汇层面的重叠情况。

    # ROUGE-2（Bigram ROUGE）：
    # ROUGE-2考虑的是双词组（bigrams）的匹配情况。它计算生成文本和参考答案中相同的双词组数量的比例，从而衡量了更长的短语匹配情况。

    # ROUGE-L（Longest Common Subsequence ROUGE）：
    # ROUGE-L考虑的是最长公共子序列（LCS）的匹配情况。它衡量了生成文本和参考答案中最长连续子序列的长度，即最长的重叠部分。ROUGE-L更关注句子结构和连贯性。

    # ROUGE-W（Weighted ROUGE）：
    # ROUGE-W考虑了多个n-gram范围，例如ROUGE-1、ROUGE-2和ROUGE-3。它计算每个范围的F1分数，并使用权重进行组合，以综合考虑不同长度的短语匹配。
'''
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
reference = "This is a reference sentence for testing."
candidate = "This is a candidate sentence for testing."
scores = scorer.score(reference, candidate)
print("ROUGE-L Precision:", scores['rougeL'].precision)
print("ROUGE-L Recall:", scores['rougeL'].recall)
print("ROUGE-L F1:", scores['rougeL'].fmeasure)


from bert_score import score

# 参考答案列表
references = ["This is a reference sentence for testing."]
# 生成文本列表
candidates = ["This is a candidate sentence for testing."]

# 计算BERT Score
P, R, F1 = score(candidates, references, lang="en")

# 打印BERT Score
print("BERT Score Precision:", P.mean().item())
print("BERT Score Recall:", R.mean().item())
print("BERT Score F1:", F1.mean().item())


import torch
from transformers import AutoTokenizer, AutoModel
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
reference = "This is a reference sentence for testing."
candidate = "This is a candidate sentence for testing."
input_ids_ref = torch.tensor(tokenizer.encode(reference, add_special_tokens=True)).unsqueeze(0)
input_ids_cand = torch.tensor(tokenizer.encode(candidate, add_special_tokens=True)).unsqueeze(0)

with torch.no_grad():
    outputs_ref = model(input_ids_ref)
    embeddings_ref = outputs_ref.last_hidden_state[:, 0, :]  # 使用 [CLS] 标记的隐藏状态
    outputs_cand = model(input_ids_cand)
    embeddings_cand = outputs_cand.last_hidden_state[:, 0, :]  # 使用 [CLS] 标记的隐藏状态

cosine_similarity = torch.nn.functional.cosine_similarity(embeddings_ref, embeddings_cand)
print("Cosine Similarity:", cosine_similarity.item())
