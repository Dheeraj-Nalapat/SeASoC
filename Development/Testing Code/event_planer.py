class EventPlanner:
    def __init__(self):
        self.events = []

    def add_event(self, event_name, event_date):
        event = {'name': event_name, 'date': event_date}
        self.events.append(event)
        print(f"Event '{event_name}' added on {event_date}.")

    def view_events(self):
        if not self.events:
            print("No events found.")
        else:
            print("Upcoming Events:")
            for idx, event in enumerate(self.events, start=1):
                print(f"{idx}. {event['name']} - {event['date']}")

    def delete_event(self, event_index):
        if 1 <= event_index <= len(self.events):
            deleted_event = self.events.pop(event_index - 1)
            print(f"Event '{deleted_event['name']}' on {deleted_event['date']} deleted.")
        else:
            print("Invalid event index.")

def main():
    event_planner = EventPlanner()

    while True:
        print("Event Planner Menu:")
        print("1. Add Event")
        print("2. View Events")
        print("3. Delete Event")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            name = input("Enter event name: ")
            date = input("Enter event date (YYYY-MM-DD): ")
            event_planner.add_event(name, date)
        elif choice == '2':
            event_planner.view_events()
        elif choice == '3':
            index = int(input("Enter the index of the event to delete: "))
            event_planner.delete_event(index)
        elif choice == '4':
            print("Exiting Event Planner. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()