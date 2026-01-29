# %%
import os
from typing import Optional

from obspy import read, Stream, UTCDateTime
import pandas as pd
from datetime import datetime, timedelta

from obspy.signal.filter import bandpass
from scipy.integrate import cumulative_trapezoid
import numpy as np
from eruption_forecast.utils import detect_outliers
from multiprocessing import Pool

# %%
sds_dir = r"D:\Data\OJN"
network = "VG"
channel = "EHZ"
station = "OJN"
channel_type = "D"
location = "00"
start_datetime = "2025-01-01"
end_datetime = "2025-01-31"
extends_data = True
tmp_dir = r"D:\Projects\eruption-forecast\output\forecast\VG.OJN.00.EHN\tremor\tmp"
tremor_dir = r"D:\Projects\eruption-forecast\output\forecast\VG.OJN.00.EHN\tremor"
verbose = False
# %%
os.makedirs(tmp_dir, exist_ok=True)
# %%
freq_bands = [(0.01, 0.1), (0.1, 2), (2, 5), (4.5, 8), (8, 16)]
band_names = ["rsam_vlf", "rsam_lf", "rsam", "rsam_mf", "rsam_hf"]
ratio_names = ["dsar_vlf_lf", "dsar_lf_rsam", "dsar_rsam_mf", "dsar_mf_hf"]
# %%
start_datetime_obj = datetime.strptime(start_datetime, "%Y-%m-%d")
end_datetime_obj = datetime.strptime(end_datetime, "%Y-%m-%d")
n_days = (end_datetime_obj - start_datetime_obj).days


# %%
def integrate(
    data: np.ndarray,
    sampling_rate: float = 100.0,
    anchor_start_sample: Optional[int] = None,
) -> np.ndarray:
    displacements: np.ndarray = cumulative_trapezoid(
        data, dx=1.0 / sampling_rate, initial=0
    )

    if verbose:
        print(
            f"displacements[{anchor_start_sample}]: {displacements[anchor_start_sample]}"
        )

    # anchoring to start_sample
    if anchor_start_sample is not None:
        displacements = displacements - displacements[anchor_start_sample]

        if verbose:
            print(
                f"after displacements[{anchor_start_sample}]: {displacements[anchor_start_sample]}"
            )

    return displacements


# %%
def load_mseed(utc_start_datetime: UTCDateTime, extends: bool = False) -> Stream:
    stream = Stream()
    extends = [-1, 0, 1] if extends else [0]

    for extend in extends:
        utc_datetime: UTCDateTime = utc_start_datetime + timedelta(days=extend)
        year, julian_day = utc_datetime.format_seed().split(",")
        filename = f"{network}.{station}.{location}.{channel}.{channel_type}.{year}.{julian_day}"

        miniseed_file = os.path.join(
            sds_dir,
            str(year),
            network,
            station,
            f"{channel}.{channel_type}",
            filename,
        )

        if not os.path.isfile(miniseed_file):
            continue

        try:
            stream = stream + read(miniseed_file, format="MSEED")
        except Exception as e:
            continue

    try:
        stream = stream.merge(method="interpolate")
        if len(stream) > 0:
            stream = stream.detrend(method="demean")
    except Exception as e:
        return stream

    return stream


# %%
def wrapped_indices(outlier_index, asymmetric_factor, subdomain_range, total_windows):
    asymmetric_factor_value = np.floor(asymmetric_factor * subdomain_range)
    start_index = int(
        outlier_index - asymmetric_factor_value
    )  # Compute the index of the domain where the subdomain centered on the peak begins

    end_index = start_index + subdomain_range  # Find the end index of the subdomain

    if end_index >= total_windows:  # If end index exceeds data range
        index = list(
            range(end_index - total_windows)
        )  # Wrap domain so continues from beginning of data range
        end = list(range(start_index, total_windows))
        index.extend(end)
    elif start_index < 0:  # If starting index exceeds data range
        index = list(range(end_index))
        end = list(
            range(total_windows + start_index, total_windows)
        )  # Wrap domains so continues at end of data range
        index.extend(end)
    else:
        index = list(range(start_index, end_index))

    if verbose:
        print(
            f"outlier_index, asymmetric_factor_valus, start_index, end_index, total_windows, len(index): {outlier_index}, {asymmetric_factor_value}, {start_index}, {end_index}, {total_windows}, {len(index)}"
        )
    # outlier_index, asymmetric_factor_valus, start_index, end_index, total_windows, len(index):
    # 7932, 3.0, 7929, 7965, 144, 7821

    return index


# %%
def get_data_for_day(
    day_index, utc_start_datetime: UTCDateTime, _station, extends: bool = False
):
    # t0 = utc_datetime
    # recalculate based on day_index
    utc_start_datetime: UTCDateTime = utc_start_datetime + timedelta(days=day_index)
    start_date_str = utc_start_datetime.strftime("%Y-%m-%d")

    print(start_date_str)

    # Check if tmp file exists
    filename = f"{start_date_str}.csv"
    filepath = os.path.join(tmp_dir, filename)
    if os.path.isfile(filepath):
        return filepath

    # Load mseed
    stream = load_mseed(utc_start_datetime, extends=extends)
    trace = stream[0]
    data = trace.data

    if verbose:
        print(f"utc_start_datetime, len_data: {utc_start_datetime}, {len(data)}")

    if len(data) == 0:
        return None

    trace_starttime = trace.stats.starttime
    sampling_rate = trace.stats.sampling_rate

    # iO = start_sample
    start_sample = utc_start_datetime - trace_starttime
    start_sample = int(start_sample * sampling_rate)

    # i1 = end_sample
    end_sample = int(24 * 3600 * sampling_rate)
    if (start_sample + end_sample) > len(data):
        end_sample = len(data)
    else:
        end_sample = start_sample + end_sample
    total_samples = end_sample - start_sample

    if verbose:
        print(
            f"day_index, start_sample, end_sample: {day_index}, ({start_sample} - {end_sample})"
        )

    # N = ten_minutes_samples
    # ten_minutes_samples = sampling_rate * 10 minutes * 60 seconds
    ten_minutes_samples = int(10 * 60 * sampling_rate)

    # m = total_windows
    total_windows = int(total_samples // ten_minutes_samples)

    if verbose:
        print(
            f"ten_minutes_samples, total_windows: {ten_minutes_samples}, {total_windows}"
        )

    # integrating
    displacement = integrate(
        data, sampling_rate=sampling_rate, anchor_start_sample=start_sample
    )

    # save temporary results
    datas = []
    columns = []

    # apply filter
    all_data = []
    all_displacement = []
    for freq_min, freq_max in freq_bands:
        _all_data = (
            abs(
                bandpass(data, freq_min, freq_max, sampling_rate)[
                    start_sample:end_sample
                ]
            )
            * 1.0e9
        )
        _all_displacement = (
            abs(
                bandpass(displacement, freq_min, freq_max, sampling_rate)[
                    start_sample:end_sample
                ]
            )
            * 1.0e9
        )

        all_data.append(_all_data)
        all_displacement.append(_all_displacement)

    if verbose:
        print(f"len(all_data): {len(all_data)}")

    # find outliers
    outliers = []
    outlier_indices = []
    for window_index in range(total_windows):
        outlier, outlier_index, _ = detect_outliers(
            all_data[2][
                window_index * ten_minutes_samples : (window_index + 1)
                * ten_minutes_samples
            ]
        )
        outliers.append(outlier)
        outlier_indices.append(outlier_index)

    if verbose:
        print(f"len_outliers: {len(outliers)}")

    # calculate RSAM
    asymmetric_factor = 0.1
    number_subdomains = 4
    subdomain_range = total_windows // number_subdomains

    for _all_data, band_name in zip(all_data, band_names):
        rsam = []
        rsam_without_outliers = []
        for window_index, outlier, outlier_index in zip(
            range(total_windows), outliers, outlier_indices
        ):
            _rsam = _all_data[
                window_index * ten_minutes_samples : (window_index + 1)
                * ten_minutes_samples
            ]
            rsam.append(np.mean(_rsam))

            if outlier:
                _outlier_index = wrapped_indices(
                    outlier_index, asymmetric_factor, subdomain_range, total_windows
                )
                _rsam = np.delete(_rsam, _outlier_index)
            rsam_without_outliers.append(np.mean(_rsam))

        datas.append(np.array(rsam))
        columns.append(band_name)

        datas.append(np.array(rsam_without_outliers))
        columns.append(f"{band_name}_outlier")

    # calculate DSAR
    for ratio_index, ratio_name in enumerate(ratio_names):
        dsar = []
        dsar_without_outliers = []
        for window_index, outlier, outlier_index in zip(
            range(total_windows), outliers, outlier_indices
        ):
            first_domain = all_displacement[ratio_index][
                window_index * ten_minutes_samples : (window_index + 1)
                * ten_minutes_samples
            ]
            second_domain = all_displacement[ratio_index + 1][
                window_index * ten_minutes_samples : (window_index + 1)
                * ten_minutes_samples
            ]

            dsar.append(np.mean(first_domain) / np.mean(second_domain))

            if outlier:
                _outlier_index = wrapped_indices(
                    outlier_index, asymmetric_factor, subdomain_range, total_windows
                )
                first_domain = np.delete(first_domain, _outlier_index)
                second_domain = np.delete(second_domain, _outlier_index)

            dsar_without_outliers.append(np.mean(first_domain) / np.mean(second_domain))

        datas.append(np.array(dsar))
        columns.append(ratio_name)

        datas.append(np.array(dsar_without_outliers))
        columns.append(f"{ratio_name}_outlier")

    datas = np.array(datas)
    index = pd.date_range(
        utc_start_datetime.datetime,
        utc_start_datetime.datetime + timedelta(days=1),
        freq="10min",
        inclusive="left",
    )
    df = pd.DataFrame(zip(*datas), columns=columns, index=index)
    df.index.name = "time"
    df.to_csv(filepath, index=True)

    print(filepath)

    return filepath


# %%
def merge_csv():
    dfs = []
    dates = pd.date_range(start_datetime_obj, end_datetime_obj, freq="D")
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{date_str}.csv"
        filepath = os.path.join(tmp_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            df = df.loc[~df.index.duplicated(keep="last")]
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=False)
    df.index.name = "time"
    df = df.loc[~df.index.duplicated(keep="last")]
    df.to_csv(os.path.join(tremor_dir, f"OJN_tremor_data.csv"), index=True)


# %%
def main(n_jobs: int = 1):
    files = []

    jobs = [
        [job_index, UTCDateTime(start_datetime_obj), "OJN.EHN.VG.00", extends_data]
        for job_index in range(n_days)
    ]

    if n_jobs == 1:
        for job in jobs:
            filepath = get_data_for_day(*job)
            if filepath:
                files.append(filepath)
    else:
        print(f"n_jobs: {n_jobs}")
        p = Pool(n_jobs)
        p.starmap(get_data_for_day, jobs)
        p.close()
        p.join()


# %%
if __name__ == "__main__":
    main(n_jobs=1)
    merge_csv()
