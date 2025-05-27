import pandas as pd
from pm4py.objects.log.util import dataframe_utils

def rename_repeating_events(df, epsilon=3, final_states=None):
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    df = df.sort_values(by=["case:concept:name", "time:timestamp"])

    updated_final_states = set()

    def process_case(case_df):
        new_names = []
        event_counts = {}
        for name in case_df["concept:name"]:
            if name not in event_counts:
                event_counts[name] = 1
            elif event_counts[name] < epsilon:
                event_counts[name] += 1
            new_names.append(f"{name}{event_counts[name]}")

        case_df["concept:name"] = new_names
        return case_df

    df = df.groupby("case:concept:name", group_keys=False).apply(process_case)
    unique_events = set(df["concept:name"].unique())
    for event in unique_events:
        for fe in final_states:
            if event.startswith(fe):
                updated_final_states.add(event)
    if final_states is not None:
        return df, sorted(updated_final_states)
    return df