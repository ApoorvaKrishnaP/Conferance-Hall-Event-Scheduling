# Conference Hall Event Scheduler

The **Conference Hall Event Scheduler** is a Streamlit-based web application designed to help users efficiently schedule non-overlapping events in a conference hall. The app provides an interactive interface for adding, deleting, and scheduling events, and visually compares the performance of different scheduling algorithms.

## Features

- **Add Events:** Input start and end times for events in `HH.MM` format.
- **Delete Events:** Remove events by specifying their event number.
- **Schedule Events:** Compute the optimal schedule using a greedy algorithm to maximize the number of non-overlapping events.
- **View Results:** See scheduled and unscheduled events in a tabular format.
- **Clear All Events:** Reset the event list with a single click.
- **Algorithm Comparison:** Benchmark and visualize the runtime of Greedy, Dynamic Programming, and Brute Force algorithms for event scheduling.

## Algorithms Implemented

- **Greedy Algorithm:** Efficiently selects the maximum number of non-overlapping events.
- **Dynamic Programming:** Finds the optimal schedule using a DP approach.
- **Brute Force:** Explores all possible subsets (for small input sizes only).

## How to Run

1. **Install Dependencies:**
    ```bash
    pip install streamlit pandas plotly
    ```

2. **Clone the Repository:**
    ```bash
    git clone https://github.com/ApoorvaKrishnaP/Conference-Hall-Event-Scheduling.git
    cd Conference-Hall-Event-Scheduling
    ```

3. **Run the Application:**
    ```bash
    streamlit run Conference_Hall_Event_Scheduler.py
    ```

4. **(Optional) Add Custom Styles:**
    - Place your `styles.css` file in the project directory for custom UI styling.

## Usage

- Enter event start and end times in the format `HH.MM` (e.g., `09.30` for 9:30 AM).
- Click **Add Event** to add it to the schedule.
- Use **Delete Event** to remove an event by its number.
- Click **Compute** to see the optimal schedule.
- Use **Run Algorithms** to benchmark and compare algorithm runtimes.

