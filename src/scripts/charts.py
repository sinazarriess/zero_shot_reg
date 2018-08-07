# Credit: Josh Hemann

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

n_types = 4
n_values = 4

colors = ['#332288', '#44AA99', '#117733', '#882255']
## first exp
type_1_values = (7.05, 16.38, 37.04, 45.4)
type_2_values = (4.59, 13.43, 36.7,46.55)
type_3_values = (10.6, 23.86, 40.53, 46.05)
type_4_values = (14.35, 21.22, 33.49, 40.24)
labels = ['Top 5 candidates', 'Top 5 candidates \n(POS-Tagger)','Top 10 candidates', 'Top 100 candidates']

## search spaces
#type_1_values = (4.72,14.71, 34,42.24)
#type_2_values = (10.6,22.72,38.74,44.75)
#type_3_values = (14.35,21.02,33.39,39.35)
#labels = ['Unrestricted search space\n(top 5 candidates)', 'Unrestricted search space\n(top 10 candidates)',
#          'Unrestricted search space\n(top 100 candidates)']

## all
#labels = ['Top 10 - \nonly noun candidates', 'Top 10-\nrestricted search', 'Top 10 - \nunrestricted search ']
#type_1_values = (17.56, 34.91, 49.21, 69.51)
#type_2_values = (10.6, 23.86, 40.53, 46.05)
#type_3_values = (10.6, 22.72, 38.74, 44.75)

ticks = ['Hit@1', 'Hit@2', 'Hit@5', 'Hit@10']

all_values = [type_1_values, type_2_values, type_3_values, type_4_values]

fig, ax = plt.subplots()

index = np.arange(n_values)
bar_width = 1 / float(n_types + 1)

opacity = 0.8
error_config = {'ecolor': '0.3'}

def make_bar(offset, values_, color_, label_):
    return ax.bar(index + offset, values_, bar_width,
            alpha=opacity, color=color_,
            error_kw=error_config,
            label=label_)

def autolabel(i, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for index, rect in enumerate(rects):
        height = rect.get_height()
        #ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
         #       '%d' % int(height),
         #       ha='center', va='bottom')
        percentage = str(round(all_values[i][index])) + " %"
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                percentage, ha='center', va='bottom')

for i in range(n_types):
    rects = make_bar(i * bar_width, all_values[i], colors[i], labels[i])
    #autolabel(i, rects)


#ax.set_xlabel('Group')
ax.set_ylabel('Accuracy in %')
ax.set_title('')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(ticks)
box = ax.get_position()
lgd = plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
#plt.legend()

plt.tight_layout()
plt.show()
fig.savefig('image_output.png', bbox_extra_artists=(lgd,), bbox_inches='tight')