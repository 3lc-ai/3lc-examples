from __future__ import annotations

from typing import Literal, cast

import tlc


class InstanceConfig:
    """Configuration class for instance handling arguments."""

    def __init__(
        self,
        instance_column: str,
        instance_type: Literal["bounding_boxes", "segmentations"],
        label_column_path: str | None = None,
        allow_label_free: bool = False,
    ):
        """Initialize instance configuration.

        :param instance_column: Name of the column containing instances (e.g., "bbs", "segmentations")
        :param instance_type: Type of instances
        :param label_column_path: Path to labels within instance column (e.g., "bbs.bb_list.label")
        :param allow_label_free: Whether to allow processing without labels
        """
        self.instance_column = instance_column
        self.instance_type = instance_type
        self.label_column_path = label_column_path
        self.allow_label_free = allow_label_free
        self._validated_tables: set[str] = set()  # Cache for validated table URLs

    @property
    def instance_properties_column(self) -> str:
        """Get the instance properties column.

        Uses predictable constants based on instance type, which is safe
        given the standardized schema structure.
        """
        if self.instance_type == "bounding_boxes":
            return "bb_list"
        elif self.instance_type == "segmentations":
            return "instance_properties"
        else:
            raise ValueError(f"Unknown instance type: {self.instance_type}")

    def _ensure_validated_for_table(self, input_table: tlc.Table) -> None:
        """Ensure this config is validated for the given table (cached).

        This method implements lazy validation - it only validates once per table
        and caches the result. This prevents redundant validation when the same
        config is used across multiple components (dataset, metrics, training).

        :param input_table: The table to validate against
        :raises ValueError: If validation fails
        """
        table_id = input_table.url
        if table_id not in self._validated_tables:
            self._validate(input_table)
            self._validated_tables.add(table_id)

    @classmethod
    def resolve(
        cls,
        input_table: tlc.Table,
        instance_column: str | None = None,
        instance_type: Literal["bounding_boxes", "segmentations", "auto"] = "auto",
        allow_label_free: bool = False,
    ) -> InstanceConfig:
        """Resolve and validate instance configuration for a table.

        This is the main entry point for instance configuration. It will:
        1. Auto-detect instance column if not provided
        2. Auto-detect instance type if set to "auto"
        3. Intelligently resolve label column path
        4. Validate the final configuration

        :param input_table: The table to configure for
        :param instance_column: Instance column name (auto-detect if None)
        :param instance_type: Instance type ("auto" to detect)
        :param allow_label_free: Whether to allow label-free operation
        :return: Resolved and validated InstanceConfig
        :raises ValueError: If configuration cannot be resolved or is invalid
        """
        # Step 1: Detect instance column if needed
        if instance_column is None:
            instance_column, detected_type = cls._detect_instance_column_and_type(input_table, instance_type)
            instance_type = detected_type

        # Ensure we have a concrete type (not "auto")
        if instance_type == "auto":
            raise ValueError(
                f"Unable to auto-infer instance column from table {input_table.name}. Set up an `InstanceConfig` "
                "manually and set `instance_type` to 'bounding_boxes' or 'segmentations'."
            )
        assert instance_type in ["bounding_boxes", "segmentations"]
        instance_type = cast(Literal["bounding_boxes", "segmentations"], instance_type)

        # Step 2: Resolve label column path
        resolved_label_path = cls._resolve_label_column_path(
            instance_column=instance_column, instance_type=instance_type
        )

        # Step 3: Create and validate configuration
        config = cls(
            instance_column=instance_column,
            instance_type=instance_type,
            label_column_path=resolved_label_path,
            allow_label_free=allow_label_free,
        )

        # Step 4: Validate the configuration
        config._validate(input_table)

        return config

    @staticmethod
    def _resolve_label_column_path(
        instance_column: str,
        instance_type: Literal["bounding_boxes", "segmentations"],
    ) -> str | None:
        """Resolve label column path based on instance type."""
        if instance_type == "bounding_boxes":
            return f"{instance_column}.bb_list.label"
        elif instance_type == "segmentations":
            return f"{instance_column}.instance_properties.label"
        else:
            raise ValueError(f"Invalid instance type: {instance_type}")

    @staticmethod
    def _detect_instance_column_and_type(
        input_table: tlc.Table,
        instance_type: Literal["bounding_boxes", "segmentations", "auto"] = "auto",
    ) -> tuple[str, Literal["bounding_boxes", "segmentations"] | None]:
        """Auto-detect the instance column in a table."""
        if instance_type == "bounding_boxes":
            if "bbs" in input_table.columns:
                return "bbs", "bounding_boxes"
            else:
                raise ValueError(f"No bounding boxes column found in table {input_table.name}")
        elif instance_type == "segmentations":
            if "segmentations" in input_table.columns:
                return "segmentations", "segmentations"
            elif "segments" in input_table.columns:
                return "segments", "segmentations"
            else:
                raise ValueError(f"No segmentation column found in table {input_table.name}")
        elif instance_type == "auto":
            if "bbs" in input_table.columns:
                return "bbs", "bounding_boxes"
            elif "segmentations" in input_table.columns:
                return "segmentations", "segmentations"
            elif "segments" in input_table.columns:
                return "segments", "segmentations"
            else:
                return "auto", None

    def _validate(self, input_table: tlc.Table) -> None:
        """Validate the instance configuration against the table schema."""
        # Validate instance column exists
        if self.instance_column not in input_table.columns:
            raise ValueError(f"Instance column '{self.instance_column}' not found in table {input_table.name}")

        # Validate instance properties column exists
        instance_schema = input_table.rows_schema.values.get(self.instance_column)
        if not instance_schema:
            raise ValueError(f"Instance column '{self.instance_column}' not found in table schema")

        properties_column = self.instance_properties_column
        if properties_column not in instance_schema.values:
            raise ValueError(
                f"Instance properties column '{properties_column}' not found in '{self.instance_column}' schema"
            )

        # Validate label column if not in label-free mode
        if not self.allow_label_free and self.label_column_path:
            # Check if label column exists by walking the path
            parts = self.label_column_path.split(".")
            current_schema = input_table.rows_schema

            for part in parts:
                if part not in current_schema.values:
                    raise ValueError(f"Label path '{self.label_column_path}' not found in table schema")
                current_schema = current_schema.values[part]
