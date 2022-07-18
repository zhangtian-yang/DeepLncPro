import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
f = open('./result.txt', 'r')
all_data = {'m':{61:[], 101:[], 141:[], 181:[],221:[], 261:[], 301:[]},
            'h':{61:[], 101:[], 141:[], 181:[],221:[], 261:[], 301:[]}}
lines = f.readlines()
# for line in lines:
#     all_metrics = line.split('\t')
#     # all_data.append(all_metrics)
#     for i in all_metrics:
#         print(type(i))
#         print(i)

for line in lines:
    metrics = line.split('\t')
    # all_data.append(all_metrics)
    all_data[metrics[0]][int(metrics[1])].append([float('%.4f' % float(metrics[3])), float('%.4f' % float(metrics[4])),
                                                  float('%.4f' % float(metrics[5])), float('%.4f' % float(metrics[6])),
                                                  float('%.4f' % float(metrics[7])), float('%.4f' % float(metrics[8]))])

lenth = [61,101,141,181,221,261,301]
m_heatmap = np.zeros((7, 7))
h_heatmap = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        m_heatmap[i, j] = all_data['m'][lenth[i]][j][1]
        h_heatmap[i, j] = all_data['h'][lenth[i]][j][1]

#mouse
f, ax = plt.subplots(figsize=(7,7))
# ax.set_title('Accuracy comparison in mouse')
ax = sns.heatmap(m_heatmap,ax=ax,vmin=0.7, vmax=0.9,annot=True,fmt='.3f', cmap="RdBu_r")
ax.set_yticklabels(['61bp','101bp','141bp','181bp','221bp','261bp','301bp'], rotation=45)
ax.set_xticklabels(['one-hot','NCP','DPCP','one-hot+NCP','one-hot+DPCP','NCP+DPCP','ALL'], rotation=45)
plt.savefig('C:\\Users\\Zhang\\Desktop\\picture\\all_lenth_encoding_heatmap_m.svg', dpi=750, bbox_inches = 'tight')

#human
f, ax = plt.subplots(figsize=(7,7))
# ax.set_title('Accuracy comparison in human')
ax = sns.heatmap(h_heatmap,ax=ax,vmin=0.7, vmax=0.9,annot=True,fmt='.3f', cmap="RdBu_r")
ax.set_yticklabels(['61bp','101bp','141bp','181bp','221bp','261bp','301bp'], rotation=45)
ax.set_xticklabels(['one-hot','NCP','DPCP','one-hot+NCP','one-hot+DPCP','NCP+DPCP','ALL'], rotation=45)
plt.savefig('C:\\Users\\Zhang\\Desktop\\picture\\all_lenth_encoding_heatmap_h.svg', dpi=750, bbox_inches = 'tight')
