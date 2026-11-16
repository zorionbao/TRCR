"""Figure 4C: ELSA deterministic restoration-resource frontiers."""
from figures.common import OKABE, subset, title, key, panel_main


def draw(ax, frame):
    data = subset(frame, 'C')
    data = data.loc[data.cohort.eq('ELSA')]
    methods = ['Joint states', 'Independent tasks', 'One task per person']
    for index, method in enumerate(methods):
        group = data.loc[data.method.eq(method)].sort_values('budget')
        ax.plot(group.budget, group.risk*100, color=OKABE[index], lw=1.25)
    interval = subset(frame, 'C uncertainty')
    interval = interval.loc[interval.cohort.eq('ELSA')].sort_values('budget')
    ax.fill_between(interval.budget, interval.lo*100, interval.hi*100,
                    color=OKABE[0], alpha=.16)
    ax.set(xlabel='Restoration units per person', ylabel='Multiple difficulty (%)',
           xlim=(0, 1.5), ylim=(0, 45))
    title(ax, 'ELSA: joint task frontier')
    key(ax, methods, OKABE[:3])


if __name__ == '__main__':
    panel_main(4, 'C', draw)
