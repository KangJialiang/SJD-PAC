def split_datalist_for_gpu(df, gpu_id, gpu_ids, node_id, node_ids):
    node_index = node_ids.index(
        node_id
    )  # Position of the current node in the node list
    gpu_index = gpu_ids.index(gpu_id)  # Position of the current GPU in the GPU list

    # first split the dataframe for different nodes
    total_nodes = len(node_ids)
    rows_per_split = len(df) // total_nodes
    start_index = node_index * rows_per_split
    end_index = (
        start_index + rows_per_split if node_index < total_nodes - 1 else len(df)
    )

    df = df[start_index:end_index]

    # then split the dataframe for different gpus
    total_gpus = len(gpu_ids)
    rows_per_split = len(df) // total_gpus
    start_index = gpu_index * rows_per_split
    end_index = start_index + rows_per_split if gpu_index < total_gpus - 1 else len(df)

    return df[start_index:end_index]


def split_dataframe_for_gpu(df, gpu_id, gpu_ids, node_id, node_ids):
    """
    Splits the dataframe for a specific GPU on a specific node, supporting arbitrary GPU and node identifiers.

    Args:
    df (pd.DataFrame): The dataframe to split.
    gpu_id (int): The identifier of the GPU for which the split is intended.
    gpu_ids (list): List of all GPU IDs across all nodes, which can be non-sequential.
    node_id (int): The identifier of the node on which the GPU is located.
    node_ids (list): List of all node IDs, which can be non-sequential.

    Returns:
    pd.DataFrame: A subset of the original dataframe intended for the specific GPU on a specific node.
    """
    # Calculate the unique index for this GPU on this node by finding its position in the global list of GPUs
    node_index = node_ids.index(
        node_id
    )  # Position of the current node in the node list
    gpu_index = gpu_ids.index(gpu_id)  # Position of the current GPU in the GPU list

    # first split the dataframe for different nodes
    total_nodes = len(node_ids)
    rows_per_split = len(df) // total_nodes
    start_index = node_index * rows_per_split
    end_index = (
        start_index + rows_per_split if node_index < total_nodes - 1 else len(df)
    )

    df = df.iloc[start_index:end_index]

    # then split the dataframe for different gpus
    total_gpus = len(gpu_ids)
    rows_per_split = len(df) // total_gpus
    start_index = gpu_index * rows_per_split
    end_index = start_index + rows_per_split if gpu_index < total_gpus - 1 else len(df)

    return df.iloc[start_index:end_index]
