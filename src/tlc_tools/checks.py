import tlc


def has_image_column(table: tlc.Table) -> bool:
    return True


def is_image_column(table: tlc.Table, column_name: str) -> bool:
    return True


def is_label_column(table: tlc.Table, column_name: str) -> bool:
    return True


def has_label_column(table: tlc.Table) -> bool:
    return True


def has_embedding_column(table: tlc.Table) -> bool:
    return True


def is_embedding_column(table: tlc.Table, column_name: str) -> bool:
    return True


def has_bb_column(table: tlc.Table) -> bool:
    return True


def is_bb_column(table: tlc.Table, column_name: str) -> bool:
    return True
