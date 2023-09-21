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

references = ["This is a reference sentence for testing."]
candidates = ["This is a candidate sentence for testing."]

P, R, F1 = score(candidates, references, lang="en")

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


def calculate_auc(y_true, y_scores):
    n = len(y_true)
    if n <= 1:
        return 0.0
    sorted_indices = sorted(range(n), key=lambda i: y_scores[i], reverse=True)
    sorted_true = [y_true[i] for i in sorted_indices]
    auc = 0.0
    num_positive = sum(sorted_true)
    num_negative = n - num_positive
    tpr = 0.0  # True Positive Rate
    fpr = 0.0  # False Positive Rate
    prev_fpr = 0.0
    prev_tpr = 0.0
    for i in range(n):
        if sorted_true[i] == 1:
            tpr += 1 / num_positive
        else:
            fpr += 1 / num_negative
        auc += 0.5 * (fpr - prev_fpr) * (tpr + prev_tpr)
        prev_fpr = fpr
        prev_tpr = tpr
    return auc

y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
auc = calculate_auc(y_true, y_scores)
print("AUC:", auc)


def calculate_average_precision(y_true, y_scores):
    n = len(y_true)
    if n <= 0:
        return 0.0

    sorted_indices = sorted(range(n), key=lambda i: y_scores[i], reverse=True)
    sorted_true = [y_true[i] for i in sorted_indices]

    num_positive = sum(sorted_true)
    if num_positive == 0:
        return 0.0

    precision_at_recall = [0.0] * n
    true_positives = [0] * n

    for i in range(n):
        if sorted_true[i] == 1:
            true_positives[i] = 1

    cumulative_precision = [0.0] * n
    cumulative_precision[0] = true_positives[0]

    for i in range(1, n):
        cumulative_precision[i] = cumulative_precision[i - 1] + true_positives[i]

    for i in range(n):
        precision_at_recall[i] = cumulative_precision[i] / (i + 1)

    average_precision = sum(precision_at_recall[i] * true_positives[i] for i in range(n)) / num_positive
    return average_precision

y_true = [1, 0, 1, 0, 1]
y_scores = [0.8, 0.2, 0.6, 0.4, 0.7]
ap = calculate_average_precision(y_true, y_scores)
print("mAP:", ap)


import numpy as np

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_intersection = max(x1_1, x1_2)
    y1_intersection = max(y1_1, y1_2)
    x2_intersection = min(x2_1, x2_2)
    y2_intersection = min(y2_1, y2_2)
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)
    
    box1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    box2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    
    return iou

def calculate_ap(precision, recall):
    mrecall = np.concatenate(([0.], recall, [1.]))
    mprecision = np.concatenate(([0.], precision, [0.]))
    
    for i in range(len(mprecision) - 2, -1, -1):
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    
    indices = np.where(mrecall[1:] != mrecall[:-1])[0] + 1
    average_precision = np.sum((mrecall[indices] - mrecall[indices - 1]) * mprecision[indices])
    
    return average_precision

def calculate_map(ground_truth_boxes, predicted_boxes, iou_threshold=0.5):
    predicted_boxes.sort(key=lambda x: x[4], reverse=True)
    
    num_ground_truth_boxes = len(ground_truth_boxes)
    true_positives = np.zeros(len(predicted_boxes))
    false_positives = np.zeros(len(predicted_boxes))
    
    for i, predicted_box in enumerate(predicted_boxes):
        iou_max = -1
        ground_truth_box_index = -1
        
        for j, ground_truth_box in enumerate(ground_truth_boxes):
            iou = calculate_iou(predicted_box[:4], ground_truth_box)
            if iou > iou_max:
                iou_max = iou
                ground_truth_box_index = j
        
        if iou_max >= iou_threshold and ground_truth_box_index >= 0:
            if ground_truth_boxes[ground_truth_box_index][-1] == 0:
                true_positives[i] = 1
                ground_truth_boxes[ground_truth_box_index][-1] = 1  # 标记该真实边界框已被匹配
            else:
                false_positives[i] = 1
        else:
            false_positives[i] = 1
    
    cumulative_true_positives = np.cumsum(true_positives)
    cumulative_false_positives = np.cumsum(false_positives)
    recall = cumulative_true_positives / num_ground_truth_boxes
    precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
    
    ap = calculate_ap(precision, recall)
    
    return ap

# 示例用法
ground_truth_boxes = [(50, 50, 150, 150, 1), (200, 200, 300, 300, 1)]
predicted_boxes = [(40, 40, 160, 160, 0.9), (200, 200, 300, 300, 0.8)]
iou_threshold = 0.5

mAP = calculate_map(ground_truth_boxes, predicted_boxes, iou_threshold)
print("mAP:", mAP)
