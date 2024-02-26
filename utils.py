

def format_relationships(rel_mapping):
    formatted_list = []
    for key, value in rel_mapping.items():
        # Convert to uppercase and replace spaces with underscores
        formatted_value = value.upper().replace(" ", "_")
        # Append the HEAD and TAIL formats to the list
        formatted_list.append(f'@{formatted_value}_HEAD@')
        formatted_list.append(f'@{formatted_value}_TAIL@')
    return formatted_list
