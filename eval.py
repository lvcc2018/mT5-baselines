import json
from utils import load_file, load_txt_file
import argparse
import sys
from rouge import Rouge
import numpy as np
import jieba
from pprint import pprint

def rouge_score(data):
    rouge_name = ["rouge-1", "rouge-2", "rouge-l"]
    item_name = ["f", "p", "r"]

    res = {}
    for name1 in rouge_name:
        for name2 in item_name:
            res["%s-%s"%(name1, name2)] = []
    for tmp_data in data:
        origin_candidate = tmp_data['candidate']
        origin_reference = tmp_data['reference']
        assert isinstance(origin_candidate, str)
        if not isinstance(origin_reference, list):
            origin_reference = [origin_reference]

        tmp_res = []
        for r in origin_reference:
            tmp_res.append(Rouge().get_scores(refs=r, hyps=origin_candidate)[0])

        for name1 in rouge_name:
            for name2 in item_name:
                res["%s-%s"%(name1, name2)].append(max([tr[name1][name2] for tr in tmp_res]))

    for name1 in rouge_name:
        for name2 in item_name:
            res["%s-%s"%(name1, name2)] = np.mean(res["%s-%s"%(name1, name2)])
    return res

def proline(lang, line):
    if lang == 'zh':
        return " ".join([w for w in line.strip()])
    else:
        return line

def evaluate(pred, gold):
    pred_data = load_txt_file(pred)
    gold_data = load_file(gold)
    assert(len(pred_data) == len(gold_data))
    res = {i:{j:{'pred':[], 'gold':[]} for j in ['en','zh','fr','es']} for i in ['SC','SUM','SPC']}
    for i in range(len(pred_data)):
        res[gold_data[i]['task']][gold_data[i]['language']]['pred'].append(pred_data[i])
        res[gold_data[i]['task']][gold_data[i]['language']]['gold'].append(gold_data[i]['target'])
    for ta in ['SC','SPC']:
        res[ta]['right'] = 0
        res[ta]['all'] = 0
        for lang in ['en','zh','fr','es']:
            if len(res[ta][lang]['gold']):
                res[ta][lang]['acc'] = sum([res[ta][lang]['gold'][i] == res[ta][lang]['pred'][i] for i in range(len(res[ta][lang]['gold']))]) / len(res[ta][lang]['gold'])
                res[ta]['right'] += sum([res[ta][lang]['gold'][i] == res[ta][lang]['pred'][i] for i in range(len(res[ta][lang]['gold']))])
                res[ta]['all'] += len(res[ta][lang]['gold'])
        if res[ta]['all'] != 0:
            res[ta]['acc'] = res[ta]['right']/res[ta]['all']
        del(res[ta]['right'], res[ta]['all'])
    all_eval_data = []
    for lang in ['en','zh','fr','es']:
        if len(res['SUM'][lang]['gold']):
            eval_data = [{"reference": [proline(lang, g)], "candidate": proline(lang, p)} for g, p in zip(res['SUM'][lang]['gold'], res['SUM'][lang]['pred'])]
            all_eval_data.extend([{"reference": [proline(lang, g)], "candidate": proline(lang, p)} for g, p in zip(res['SUM'][lang]['gold'], res['SUM'][lang]['pred'])])
            res['SUM'][lang]['rouge'] = rouge_score(eval_data)
    res['SUM']['rouge'] = rouge_score(all_eval_data)
    for i in ['SC','SUM','SPC']:
        for j in ['en','zh','fr','es']:
            del(res[i][j]['pred'])
            del(res[i][j]['gold'])
    pprint(res)

    

        

def main():
    pred = '/home/lvcc/mT5-baselines/result/mT5_XLSUM_1e-05_dict10_best_eval_noshuffle.txt'
    gold = '/home/lvcc/mT5-baselines/data/XLSUM/full/test.json'
    evaluate(pred, gold)

if __name__ == '__main__':
    main()