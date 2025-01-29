from datetime import datetime

class DailyCounter:
    def __init__(self, limit=100):
        self.limit = limit
        self.current_date = datetime.now().date()
        self.count = 0

    def increment(self):
        today = datetime.now().date()
        
        # Reset the count if the date has changed
        if today != self.current_date:
            self.current_date = today
            self.count = 0

        # Increment the count if it's below the limit
        print(f"Current Date: {self.current_date} Current Daily Call: {self.count}")
        if self.count < self.limit:
            self.count += 1
            return self.count, True
        else:
            return self.count, False