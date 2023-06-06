from densevid_eval3.evaluate2018 import main as eval2018
from densevid_eval3.evaluate2021 import main as eval2021

def eval_dvc(json_path, reference, no_lang_eval=False, topN=1000, version='2021'):
    args = type('args', (object,), {})()
    args.submission = json_path
    args.max_proposals_per_video = topN
    args.tious = [0.3,0.5,0.7,0.9]
    args.verbose = False
    args.no_lang_eval = no_lang_eval
    args.references = reference
    eval_func = eval2018 if version=='2018' else eval2021
    score = eval_func(args)
    return score

if __name__ == '__main__':
    p = 'prediction/num2471_epoch29.json_rerank_alpha1.0_temp2.0.json'
    ref = ['data/test.json']
    score = eval_dvc(p, ref, no_lang_eval=False, version='2021')
    print(score)