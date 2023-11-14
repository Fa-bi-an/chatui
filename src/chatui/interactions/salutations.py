from datetime import datetime

def get_time_based_greeting():
    """
    Returns a greeting message based on the current time of day.

    The function divides the day into four time periods:
    - Morning (5 AM to 11:59 AM): Returns "Good morning!"
    - Afternoon (12 PM to 4:59 PM): Returns "Good afternoon!"
    - Evening (5 PM to 9:59 PM): Returns "Good evening!"
    - Night (10 PM to 4:59 AM): Returns "Good night!"

    Returns:
        str: A greeting message appropriate for the current time of day.
    """
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning!"
    elif 12 <= current_hour < 17:
        return "Good afternoon!"
    elif 17 <= current_hour < 22:
        return "Good evening!"
    else:
        return "Good night!"





if __name__ == "__main__":
    # Using the function
    greeting = get_time_based_greeting()
    print(greeting)
