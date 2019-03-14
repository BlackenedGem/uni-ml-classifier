# This file contains functions used in the coursework
# It's easier to separate them out for readability

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
