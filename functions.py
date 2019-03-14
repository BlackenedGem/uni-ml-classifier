# This file contains functions used in the coursework
# It's easier to separate them out for readability

def format_time(start_time, end_time):
    time_diff = end_time - start_time

    seconds = time_diff // 60
    time_diff -= seconds
    time_diff //= 60

    minutes = time_diff // 60
    time_diff -= minutes

    hours = time_diff // 60

    time_string = f"{seconds}s"
    if minutes > 0:
        time
    pass