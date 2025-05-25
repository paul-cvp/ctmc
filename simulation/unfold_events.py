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
            else:
                event_counts[name] += 1

            count = event_counts[name]
            if count <= epsilon:
                new_name = name if count == 1 else f"{name}{count}"
            else:
                new_name = f"{name}{epsilon}"

            new_names.append(new_name)

        case_df["concept:name"] = new_names

        if final_states:
            last_event = new_names[-1]
            original_last_event = case_df["concept:name"].iloc[-1].split(str(epsilon))[0].rstrip("1234567890")
            if original_last_event in final_states:
                updated_final_states.add(last_event)

        return case_df

    df = df.groupby("case:concept:name", group_keys=False).apply(process_case)

    if final_states is not None:
        return df, sorted(updated_final_states)
    return df