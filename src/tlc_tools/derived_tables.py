import tlc

from tlc_tools.common import check_is_segmentation_column


def masks_to_polygons(
    table: tlc.Table,
    segmentation_column: str = tlc.SEGMENTATIONS,
    output_table_name: str = "polygons",
) -> tlc.Table:
    """Convert a table of instance segmentation masks to a table of instance segmentation polygons.

    :param table: The table to convert.
    :param segmentation_column: The name of the column containing the segmentation masks.
    :param output_table_name: The name of the output table.
    :return: A table of instance segmentation polygons.
    """
    check_is_segmentation_column(table, segmentation_column, "instance_segmentation_masks")
    polygon_table = tlc.EditedTable(
        url=table.url.create_sibling(output_table_name).create_unique(),
        input_table_url=table.url,
        override_table_rows_schema={
            "values": {
                segmentation_column: {
                    "sample_type": tlc.InstanceSegmentationPolygons.sample_type,
                },
            },
        },
    )

    polygon_table.write_to_url()

    return polygon_table


def polygons_to_masks(
    table: tlc.Table,
    segmentation_column: str = tlc.SEGMENTATIONS,
    output_table_name: str = "masks",
) -> tlc.Table:
    """Convert a table of instance segmentation polygons to a table of instance segmentation masks.

    :param table: The table to convert.
    :param segmentation_column: The name of the column containing the segmentation polygons.
    :param output_table_name: The name of the output table.
    :return: A table of instance segmentation masks.
    """
    check_is_segmentation_column(table, segmentation_column, "instance_segmentation_polygons")
    mask_table = tlc.EditedTable(
        url=table.url.create_sibling(output_table_name).create_unique(),
        input_table_url=table.url,
        override_table_rows_schema={
            "values": {
                segmentation_column: {
                    "sample_type": tlc.InstanceSegmentationMasks.sample_type,
                },
            },
        },
    )

    mask_table.write_to_url()

    return mask_table
