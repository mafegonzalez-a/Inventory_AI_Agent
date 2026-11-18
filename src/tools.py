import pandas as pd

def filter_information (dataframe: pd.DataFrame, column: str, operator:str, value: str|int|float) -> pd.DataFrame:
    """Filter an inventory DataFrame by comparing a column to a value.

    Args:
        dataframe: Inventory data to filter.
        column: Name of the column to compare.
        operator: Comparison operator. Supported values are ``==``, ``!=``,
            ``>``, ``<``, ``>=``, and ``<=``.
        value: Value to compare against the column.

    Returns:
        A DataFrame containing only the rows that satisfy the comparison.

    Raises:
        ValueError: If ``operator`` is not one of the supported operators.
        KeyError: If ``column`` does not exist in ``dataframe``.
    """
    if operator == "==":
        return dataframe[dataframe[column] == value]
    elif operator == "!=":
        return dataframe[dataframe[column] != value]
    elif operator == ">":
        return dataframe[dataframe[column] > value]
    elif operator == "<":
        return dataframe[dataframe[column] < value]
    elif operator == ">=":
        return dataframe[dataframe[column] >= value]
    elif operator == "<=":
        return dataframe[dataframe[column] <= value]
    else:
        raise ValueError(f"Invalid operator: {operator}")

def get_information(query: dict) -> list[dict]:
    """Query product information from the inventory CSV file.

    The method reads ``data/products.csv`` relative to the current working
    directory, then applies the requested filters and transformations in the
    following order: filtering, aggregation, column selection, sorting, and
    row limiting.

    Args:
        query: Dictionary describing the requested inventory operation. Supported
            keys are:

            * ``filters``: Optional list of dictionaries containing ``column``,
              ``operator``, and ``value``.
            * ``group_by``: Optional column name used to group records.
            * ``operation``: Aggregation operation accepted by pandas, such as
              ``"sum"``, ``"mean"``, ``"count"``, or ``"max"``.
            * ``select``: Optional list of column names to include in the result.
            * ``sort_by``: Optional column name used to sort the result.
            * ``limit``: Optional integer limiting the number of returned records.

    Returns:
        A list of dictionaries representing the resulting rows. Each dictionary
        maps a result column name to its value.

    Raises:
        ValueError: If a filter uses an unsupported operator.
        KeyError: If a filter references a column that is not present, or if a
            filter, selected-column, or sorting key is missing the required
            structure.
        FileNotFoundError: If ``data/products.csv`` cannot be found.
    """
    inventory_data = pd.read_csv("data/products.csv")

    for filter in query.get("filters", []):
        column = filter["column"]
        operator = filter["operator"]
        value = filter["value"]

        inventory_data = filter_information(inventory_data, column, operator, value)

    # Select columns for aggregation
    selected_columns = query.get("select")

    if selected_columns:
        missing_columns = [
            column for column in selected_columns
            if column not in inventory_data.columns
        ]

        if missing_columns:
            raise KeyError(f"Columns not found: {missing_columns}")

    # Apply aggregation
    operation = query.get("operation")
    group_by = query.get("group_by")

    if operation:
        if not selected_columns:
            raise ValueError(
                "Columns must be specified for aggregation."
            )

        if operation not in {"sum", "mean", "count", "max", "min"}:
            raise ValueError(f"Unsupported operation: {operation}")

        if operation in {"sum", "mean"}:
            for column in selected_columns:
                if not pd.api.types.is_numeric_dtype(
                    inventory_data[column]
                ):
                    raise TypeError(
                        f"Column '{column}' must be numeric."
                    )

        if group_by:
            if group_by not in inventory_data.columns:
                raise KeyError(f"Column not found: {group_by}")

            inventory_data = (
                inventory_data
                .groupby(group_by)[selected_columns]
                .agg(operation)
                .reset_index()
            )

        else:
            inventory_data = (
                inventory_data[selected_columns]
                .agg(operation)
                .to_frame()
                .T
            )

    elif selected_columns:
        inventory_data = inventory_data[selected_columns]

    # Sort results
    sort_by = query.get("sort_by")

    if sort_by:
        if sort_by not in inventory_data.columns:
            raise KeyError(f"Column not found: {sort_by}")

        inventory_data = inventory_data.sort_values(by=sort_by)

    # Limit results
    limit = query.get("limit")

    if limit is not None:
        inventory_data = inventory_data.head(limit)

    return inventory_data.to_dict(orient="records")
