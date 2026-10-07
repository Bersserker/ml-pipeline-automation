from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe

DATA_PATH = Path("data/raw/UCI_Credit_Card.csv")


def main():
    context = gx.get_context()

    data_source = context.data_sources.add_pandas_filesystem(
        name="credit_data_source",
        base_directory="data/raw",
    )

    asset = data_source.add_csv_asset(
        name="credit_data",
    )

    batch_definition = asset.add_batch_definition_path(
        name="credit_batch",
        path=DATA_PATH.name,
    )

    suite = gx.ExpectationSuite(name="credit_suite")

    suite.add_expectation(
        gxe.ExpectColumnValuesToNotBeNull(
            column="age",
        )
    )

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="age",
            min_value=18,
            max_value=100,
        )
    )

    suite = context.suites.add(suite)

    validation_definition = gx.ValidationDefinition(
        name="credit_validation",
        data=batch_definition,
        suite=suite,
    )

    validation_definition = context.validation_definitions.add(validation_definition)

    result = validation_definition.run()

    if not result.success:
        raise RuntimeError("Great Expectations validation failed")

    print("Great Expectations validation passed")


if __name__ == "__main__":
    main()
