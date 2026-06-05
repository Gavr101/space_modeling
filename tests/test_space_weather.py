from datetime import datetime, timezone

import numpy as np

from dynamics.space_weather import (
    parse_celestrak_space_weather_csv,
    quiet_space_weather_sample,
    sample_space_weather,
)


SAMPLE_CSV = (
    "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,"
    "AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,"
    "F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,"
    "F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n"
    "2024-01-06,0,0,1,1,1,1,1,1,1,1,8,1,1,1,1,1,1,1,1,1,"
    "0.0,0,41,141.0,139.0,OBS,140.0,138.0,142.0,137.0\n"
    "2024-01-07,0,0,1,1,1,1,1,1,1,1,8,2,3,4,5,6,7,8,9,6,"
    "0.0,0,42,151.0,149.0,OBS,150.0,148.0,152.0,147.0\n"
)


def test_parse_celestrak_space_weather_csv_and_sample_3h_ap() -> None:
    records = parse_celestrak_space_weather_csv(SAMPLE_CSV)
    sample = sample_space_weather(
        datetime(2024, 1, 7, 10, 15, tzinfo=timezone.utc),
        records,
    )

    assert sample.f107 == 139.0
    assert sample.f107a == 152.0
    assert sample.ap == 5.0
    assert sample.ap_daily == 6.0
    assert sample.source == "celestrak:2024-01-07:ap4"


def test_space_weather_missing_date_uses_explicit_quiet_fallback() -> None:
    records = parse_celestrak_space_weather_csv(SAMPLE_CSV)
    sample = sample_space_weather(
        datetime(2030, 1, 1, tzinfo=timezone.utc),
        records,
    )
    quiet = quiet_space_weather_sample()

    assert sample == quiet
    np.testing.assert_allclose([sample.f107a, sample.f107, sample.ap], [150.0, 150.0, 4.0])
