def parse_simin_dict(simin_dict):
    # simin dict is looks like this {label: difficulty}
    assert len(simin_dict) == 1, simin_dict
    return list(simin_dict.items())[0]

def parse_multi_simin_dict(simin_dict):
    # simin dict but with multiple keys
    pairs = list(simin_dict.items())
    return "&".join([each[0] for each in pairs]), pairs[0][1]

def parse_multi_simin_dict_as_list(simin_dict):
    # simin dict but with multiple keys
    pairs = list(simin_dict.items())
    return pairs
