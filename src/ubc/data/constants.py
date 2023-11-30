idx2label = {num: val for num, val in enumerate(["HGSC", "LGSC", "EC", "CC", "MC", "Other"])}
label2idx = {val: num for num, val in idx2label.items()}

idx2labelmask = {num: val for num, val in enumerate(["Tumor", "Stroma", "Necrosis"])}
label2idxmask = {val: num for num, val in idx2labelmask.items()}
