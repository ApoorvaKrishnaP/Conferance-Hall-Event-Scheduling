import streamlit as st
import time
import pandas as pd
import os
import random
import plotly.graph_objects as go

class Event:
    def __init__(self, start, end, index):
        self.start = start
        self.end = end
        self.index = index

# QuickSort
def partition(events, low, high):
    pivot = events[high].end
    i = low - 1
    for j in range(low, high):
        if events[j].end <= pivot:
            i += 1
            events[i], events[j] = events[j], events[i]
    events[i + 1], events[high] = events[high], events[i + 1]
    return i + 1

def quickSort(events, low, high):
    if low < high:
        pi = partition(events, low, high)
        quickSort(events, low, pi - 1)
        quickSort(events, pi + 1, high)

def sortEvents(events):
    quickSort(events, 0, len(events) - 1)

# Greedy algorithm
def scheduleEventsGreedy(events):
    selected = []
    last_end_time = -1
    for event in events:
        if event.start >= last_end_time:
            selected.append(event.index)
            last_end_time = event.end
    return selected

# Brute Force algorithm
def scheduleEventsBruteForce(events):
    n = len(events)
    max_selected = []
    
    def is_valid_subset(subset):
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                e1, e2 = subset[i], subset[j]
                if not (e1.end <= e2.start or e2.end <= e1.start):
                    return False
        return True
    
    def backtrack(index, current_subset):
        nonlocal max_selected
        if index == n:
            if is_valid_subset(current_subset) and len(current_subset) > len(max_selected):
                max_selected = [e.index for e in current_subset]
            return
        current_subset.append(events[index])
        backtrack(index + 1, current_subset)
        current_subset.pop()
        backtrack(index + 1, current_subset)
    
    backtrack(0, [])
    return max_selected

# Dynamic Programming algorithm
def scheduleEventsDP(events):
    if not events:
        return []
    n = len(events)
    sorted_events = sorted(events, key=lambda x: x.end)
    dp = [1] * n
    prev = [-1] * n
    
    for i in range(1, n):
        for j in range(i):
            if sorted_events[i].start >= sorted_events[j].end and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
    
    selected = []
    i = dp.index(max(dp))
    while i != -1:
        selected.append(sorted_events[i].index)
        i = prev[i]
    return sorted(selected, reverse=True)

# Generate random events
def generate_random_events(n):
    events = []
    for i in range(1, n + 1):
        # Random start time between 0.00 and 22.00
        start_hours = random.randint(0, 22)
        start_minutes = random.choice([0, 15, 30, 45])
        start = float(f"{start_hours}.{start_minutes:02d}")
        # Random duration between 15 minutes and 2 hours
        duration_minutes = random.randint(15, 120)
        end_hours = int(start + duration_minutes // 60)
        end_minutes = (start_minutes + duration_minutes % 60) % 60
        if end_minutes < start_minutes:
            end_hours += 1
        end = float(f"{end_hours}.{end_minutes:02d}")
        if end <= 23.59 and end > start:
            events.append(Event(start, end, i))
    return events

# Benchmark algorithms
@st.cache_data
def benchmark_algorithms(max_size, step=100, trials=3):
    sizes = list(range(100, max_size + 1, step))
    brute_sizes = list(range(10, 21, 5))  # Limit brute force to small sizes
    runtimes = {"Greedy": [], "DP": [], "Brute Force": []}
    
    for size in sizes:
        greedy_times = []
        dp_times = []
        for _ in range(trials):
            events = generate_random_events(size)
            # Greedy
            start = time.perf_counter()
            events_copy = events.copy()
            sortEvents(events_copy)
            scheduleEventsGreedy(events_copy)
            greedy_times.append(time.perf_counter() - start)
            # DP
            start = time.perf_counter()
            scheduleEventsDP(events_copy)
            dp_times.append(time.perf_counter() - start)
        runtimes["Greedy"].append(sum(greedy_times) / trials)
        runtimes["DP"].append(sum(dp_times) / trials)
    
    # Brute force on smaller sizes
    for size in brute_sizes:
        brute_times = []
        for _ in range(trials):
            events = generate_random_events(size)
            start = time.perf_counter()
            scheduleEventsBruteForce(events)
            brute_times.append(time.perf_counter() - start)
        runtimes["Brute Force"].append(sum(brute_times) / trials)
    
    return sizes, brute_sizes, runtimes

# Load CSS from file
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f"<style>{f.read()}</style>"
    return ""

css_file = "styles.css"
st.markdown(load_css(css_file), unsafe_allow_html=True)

st.title("Event Scheduler")

if 'events' not in st.session_state:
    st.session_state.events = []
    st.session_state.event_count = 0

def is_valid_time(time_str):
    try:
        hours, minutes = map(int, time_str.split("."))
        return 0 <= hours <= 23 and 0 <= minutes <= 59
    except ValueError:
        return False

st.subheader("Add Event")
col1, col2 = st.columns(2)
with col1:
    start_time_str = st.text_input("Start Time (HH.MM)", value="00.00", key="start_time")
with col2:
    end_time_str = st.text_input("End Time (HH.MM)", value="00.00", key="end_time")

if is_valid_time(start_time_str) and is_valid_time(end_time_str):
    start_hours, start_minutes = map(int, start_time_str.split("."))
    end_hours, end_minutes = map(int, end_time_str.split("."))
    start_time = float(f"{start_hours}.{start_minutes:02d}")
    end_time = float(f"{end_hours}.{end_minutes:02d}")
else:
    st.error("Invalid time format! Please enter a valid time in HH.MM format (e.g., 10.30, 22.45).")
    start_time, end_time = None, None

if st.button("Add Event", key="add_event"):
    if start_time is not None and end_time is not None:
        if end_time > start_time:
            st.session_state.event_count += 1
            st.session_state.events.append(Event(start_time, end_time, st.session_state.event_count))
            st.success(f"Event {st.session_state.event_count} added.")
        else:
            st.error("End time must be greater than start time.")

st.subheader("Current Events")
if st.session_state.events:
    df = pd.DataFrame(
        [(e.index, f"{e.start:.2f}", f"{e.end:.2f}") for e in st.session_state.events],
        columns=["Event Number", "Start Time", "End Time"]
    )
    st.dataframe(df, use_container_width=True)
else:
    st.write("No events added.")

st.subheader("Delete Event")
delete_index = st.number_input("Enter event number to delete", min_value=1, step=1, key="delete_index")
if st.button("Delete Event", key="delete_event"):
    if any(e.index == delete_index for e in st.session_state.events):
        st.session_state.events = [e for e in st.session_state.events if e.index != delete_index]
        for i, e in enumerate(st.session_state.events, 1):
            e.index = i
        st.session_state.event_count = len(st.session_state.events)
        st.success(f"Event {delete_index} deleted.")
        st.rerun()
    else:
        st.error("Invalid event number.")

st.subheader("Compute Schedule")
if st.button("Compute", key="compute"):
    if not st.session_state.events:
        st.error("No events to schedule.")
    else:
        start = time.perf_counter()
        events_copy = st.session_state.events.copy()
        sortEvents(events_copy)
        selected = scheduleEventsGreedy(events_copy)
        end = time.perf_counter()
        runtime = end - start

        selected_events = [e for e in events_copy if e.index in selected]
        def format_time(time_float):
            hours = int(time_float)
            minutes = round((time_float - hours) * 100)
            return f"{hours:02d}:{minutes:02d}"

        df_selected = pd.DataFrame(
            [(e.index, f"{format_time(e.start)} - {format_time(e.end)}") for e in selected_events],
            columns=["Event Number", "Scheduled Time"]
        )

        unscheduled_events = [e for e in events_copy if e.index not in selected]
        df_unscheduled = pd.DataFrame(
            [(e.index, f"{format_time(e.start)} - {format_time(e.end)}") for e in unscheduled_events],
            columns=["Event Number", "Unscheduled Time"]
        )

        st.write(f"Maximum number of events that can be scheduled: {len(selected)}")
        st.write(f"Selected events (by event number): {', '.join(map(str, selected))}")
        st.write(f"Runtime: {runtime:.6f} seconds")
        st.subheader("Scheduled Events")
        st.dataframe(df_selected, use_container_width=True)
        st.subheader("Unscheduled Events")
        if unscheduled_events:
            st.dataframe(df_unscheduled, use_container_width=True)
        else:
            st.write("No unscheduled events.")

if st.button("Clear All Events", key="clear_events"):
    st.session_state.events = []
    st.session_state.event_count = 0
    st.success("All events cleared.")
    st.rerun()

st.subheader("Comparison of Algorithms")
max_size = st.number_input("Maximum number of events", min_value=100, max_value=1000, value=1000, step=100, key="max_size")
if st.button("Run Algorithms", key="run_benchmark"):
    if max_size < 100:
        st.error("Maximum size must be at least 100.")
    else:
        with st.spinner("Running algorithms..."):
            sizes, brute_sizes, runtimes = benchmark_algorithms(max_size)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sizes, y=runtimes["Greedy"], mode="lines+markers", name="Greedy (O(n log n))", line=dict(color="#6ab04c")))
            fig.add_trace(go.Scatter(x=sizes, y=runtimes["DP"], mode="lines+markers", name="Dynamic Programming (O(n log n))", line=dict(color="#4a90e2")))
            fig.add_trace(go.Scatter(x=brute_sizes, y=runtimes["Brute Force"], mode="lines+markers", name="Brute Force (O(2ⁿ × n²))", line=dict(color="#e57373")))
            
            fig.update_layout(
                title="Algorithm Runtime Comparison",
                xaxis_title="Input Size (Number of Events)",
                yaxis_title="Average Runtime (Seconds)",
                yaxis_type="log", 
                showlegend=True,
                template="plotly_white",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.write("Note: Brute Force is only tested up to 20 events due to its exponential complexity.")


