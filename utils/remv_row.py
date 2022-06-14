def remv_row(df,exp_case):
    index = []
    for case in list(df["patient_id"]):
        if case in exp_case:
            index += list(df[df["patient_id"]==case].index)        
    df.drop(index, inplace = True)
    return df.reset_index(drop = True)