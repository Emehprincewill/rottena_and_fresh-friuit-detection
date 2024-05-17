import streamlit as st
import time
import psutil


def stream_data(text:str):
    for word in text.split():
        time.sleep(0.04)
        yield word + " "
        time.sleep(0.04)


def delayed_function():
    st.write_stream(stream_data)


def spinner_call(loader:str, display_text:str, spin:int) -> None:
    with st.spinner(loader):
        time.sleep(spin)
        st.success(display_text)


def remark(time: int,cpu_utilization:int,memory_utilization:int) -> str:
    remark = f"""Data migration was successful. The migration took {time/60} minutes, CPU Utilization of {cpu_utilization}%, 
                and Memory Utilization of {memory_utilization}% for the data was extraction, translation, and loading data into the MySQL database. 
                While the system has attempted to ensure validation, it may sometimes generate 
                incorrect schema or data, so it is advisable to double-check its responses in the database. 
                Thank you."""
    return remark

# Function to get CPU and memory utilization
def get_cpu_memory_usage():
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_percent = memory_info.percent
    return cpu_percent, memory_percent
    