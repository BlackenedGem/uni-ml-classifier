# This file contains functions used in the coursework
# It's easier to separate them out for readability

import matplotlib.pyplot as plt

def format_time(duration):
    seconds = duration % 60
    duration //= 60
    minutes = int(duration % 60)
    hours = int(duration // 60)

    time_string = f"{seconds:02}"
    if hours > 0:
        time_string = f"{hours}:{minutes:02}:{seconds:02.0f}"
    elif minutes > 0:
        time_string = f"{minutes:02}:{seconds:02.0f}"
    else:
        time_string = f"{seconds:.1f}s"

    return time_string

def create_end_graphs(acc, val_acc, loss, val_loss):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Accuracy")
    plt.plot(acc, 'b-', label='training')
    plt.plot(val_acc, 'g-', label='validation')
    plt.legend(loc='center right')

    plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.plot(loss, 'b-', label='training')
    plt.plot(val_loss, 'g-', label='validation')
    plt.legend(loc='lower right')

    plt.show()
