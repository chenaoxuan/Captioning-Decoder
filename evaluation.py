from nlgeval import NLGEval

nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)
all_caption_lists=[["猫 和 老 鼠","我 爱 你"],["老 鼠 和 猫","你 爱 我"]]
all_result_lists=["老 鼠 和 猫","我 爱 你"]

metrics_nlg = nlgEvalObj.compute_metrics(ref_list=all_caption_lists, hyp_list=all_result_lists)

print(">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}"
      .format(metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
print(">>>  METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}"
      .format(metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"],metrics_nlg["CIDEr"]))
