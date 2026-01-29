# Third party imports
from loguru import logger

# Project imports
from eruption_forecast import LabelBuilder


@logger.catch
def main(
    start_date: str,
    end_date: str,
    window_size: int,
    window_step: int,
    sampling_rate: int,
    day_to_forecast: int,
    eruption_dates: list[str],
    volcano_id: str,
) -> LabelBuilder:
    label_builder = LabelBuilder(
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        window_step=window_step,
        sampling_rate=sampling_rate,
        day_to_forecast=day_to_forecast,
        eruption_dates=eruption_dates,
        volcano_id=volcano_id,
        verbose=True,
    )
    label_builder.build()

    return label_builder


if __name__ == "__main__":
    eruptions = [
        "2025-03-20",
        "2025-04-22",
        "2025-05-18",
        "2025-06-17",
        "2025-07-07",
        "2025-08-01",
        "2025-08-17",
    ]

    label_builder: LabelBuilder = main(
        start_date="2025-01-01",
        end_date="2025-12-24",
        window_size=2,
        window_step=12,
        sampling_rate=100.0,
        day_to_forecast=2,
        eruption_dates=eruptions,
        volcano_id="Lewotobi Laki-laki",
    )
