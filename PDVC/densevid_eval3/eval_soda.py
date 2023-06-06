import sys
import os
from os.path import dirname, abspath

# pdvc_dir = dirname(dirname(abspath(__file__)))
# sys.path.append(pdvc_dir)

import numpy as np
from densevid_eval3.SODA.soda import SODA
from densevid_eval3.SODA.dataset import ANETCaptions
from densevid_eval3.eval_para import eval_para
import argparse

def eval_tool(prediction, referneces=None, metric='Meteor', soda_type='c', verbose=False, multireference=False):

    args = type('args', (object,), {})()
    args.prediction = prediction
    args.references = referneces
    args.metric = metric
    args.soda_type = soda_type
    args.tious = [0.3, 0.5, 0.7, 0.9]
    args.verbose = verbose

    data = ANETCaptions.from_load_files(args.references,
                                        args.prediction,
                                        multi_reference=multireference,
                                        verbose=args.verbose,
                                        )
    data.preprocess()
    if args.soda_type == 'a':
        tious = args.tious
    else:
        tious = None
    evaluator = SODA(data,
                     soda_type=args.soda_type,
                     tious=tious,
                     scorer=args.metric,
                     verbose=args.verbose
                     )
    result = evaluator.evaluate()

    return result

def eval_soda(p, ref_list,verbose=False, multireference=False):
    score_sum = []
    if multireference:
        r = eval_tool(prediction=p, referneces=ref_list, verbose=verbose, soda_type='c', multireference=True)
        score_sum.append(r['Meteor'])
    else:
        for ref in ref_list:
            r = eval_tool(prediction=p, referneces=[ref], verbose=verbose, soda_type='c')
            score_sum.append(r['Meteor'])
    print(score_sum)
    soda_avg = np.mean(score_sum, axis=0) #[avg_pre, avg_rec, avg_f1]
    soda_c_avg = soda_avg[-1]
    results = {'soda_c': soda_c_avg}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str, default='sample_submission.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('-m', '--multireference', action='store_true',
                        help='Print intermediate steps.')
    args = parser.parse_args()

    ref_list = ['data/test.json']
    score=eval_soda(args.submission, ref_list, verbose=True, multireference=args.multireference)
    print(score)
