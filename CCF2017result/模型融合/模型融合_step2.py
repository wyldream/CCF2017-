import numpy as np
import pandas as pd
from collections import Counter

path = ''

lgb = pd.read_csv(path+'ronghe_1209.csv')
gbdt = pd.read_csv(path+'gbdt6704.csv')
rf = pd.read_csv(path+'rf6662.csv')

PROB = 0.75*lgb.PROB + 0.15*gbdt.PROB + 0.1*rf.PROB
# PROB = 0.7*lgb.PROB + 0.15*gbdt.PROB + 0.15*rf.PROB
# PROB = 0.6*lgb.PROB + 0.2*gbdt.PROB + 0.2*rf.PROB
# PROB = 0.5*lgb.PROB + 0.3*gbdt.PROB + 0.2*rf.PROB
# PROB = 0.7*lgb.PROB + 0.2*gbdt.PROB + 0.1*rf.PROB
# PROB = 0.8*lgb.PROB + 0.1*gbdt.PROB + 0.1*rf.PROB

y_prob = []
for y in PROB:
    y_prob.append(y)
PROB[PROB > 0.205] = 1
PROB[~(PROB > 0.205)] = 0

# test_pos_count = PROB[PROB == 1].count()
# test_neg_count = PROB[PROB == 0].count()
# test_all_coutn = PROB.count()
# print(test_pos_count)
# print(test_neg_count)
# print(test_pos_count/test_all_coutn)
# print(test_neg_count/test_all_coutn)


submission_data = pd.DataFrame(data= {'EID':lgb.EID,
                                      'FORTARGET': PROB,
                                      'PROB':y_prob
                                      },
                               columns=['EID','FORTARGET','PROB'])
print(submission_data)
submission_data.to_csv(path_or_buf='../../output/final.csv',index=None,header=True)

