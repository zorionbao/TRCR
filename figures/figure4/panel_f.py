"""Figure 4F: CHARLS attempt-resource frontiers by restoration success."""
import numpy as np
from figures.common import OKABE, subset, title, key, panel_main


def draw(ax, frame):
    data = subset(frame, 'F')
    data = data.loc[data.cohort.eq('CHARLS')
                    & data.scenario.eq('Equal attempt costs and unrestricted task capacity')
                    & data.dependence.eq('Independent successes')
                    & data.policy.eq('Joint subset optimization')]
    for index, probability in enumerate([.25, .5, .75, 1.]):
        group = data.loc[np.isclose(data.q, probability)].sort_values('budget')
        ax.plot(group.budget, group.risk*100, color=OKABE[index], lw=1.2)
    ax.set(xlabel='Attempt units per person', ylabel='Multiple difficulty (%)',
           xlim=(0, 1.5), ylim=(0, 45))
    title(ax, 'CHARLS: restoration success')
    key(ax, ['q = 0.25', 'q = 0.50', 'q = 0.75', 'q = 1.00'], OKABE[:4])


if __name__ == '__main__':
    panel_main(4, 'F', draw)
