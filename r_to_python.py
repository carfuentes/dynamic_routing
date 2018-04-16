def get_r_dictionaries(file,mapping=True):
    with open(file, "r") as in_file:
        dict_map={}
        for line in in_file:
            line.strip()
            if mapping:
                dict_map[line.split()[1]]=line.split()[0]
            else:
                dict_map[line.split()[0]]=set(line.split()[1:])
    return dict_map


def mapping_relabel_function(net,map_dict):
    mapping_relabel = map_dict
    for node in net.nodes():
        if node not in map_dict.values():
            mapping_relabel[node]=node
    return mapping_relabel

