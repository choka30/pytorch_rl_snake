import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # Turn on interactive mode
def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')
    
    plt.ylim(ymin=0)

    plt.text(0, 0, f'Games: {len(scores)}', fontdict={'size': 20, 'color': 'red'})
    plt.text(0, 10, f'Mean Score: {mean_scores[-1]:.2f}', fontdict={'size': 20, 'color': 'blue'})

    plt.legend(loc='upper left')
    plt.show()
    plt.pause(.1)