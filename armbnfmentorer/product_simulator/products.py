import csv

import pandas as pd
import xarray as xr



def read_config(fn) -> dict:
    out = {}
    section = None
    new_section = True

    with open(fn, encoding="utf-8") as config:
        for raw_line in config:
            line = raw_line.strip()
            if not line:
                new_section = True
                continue

            if new_section:
                section = out.setdefault(line, {})
                new_section = False
                continue

            for field in line.split(";"):
                field = field.strip()
                if field.startswith("SN "):
                    section["SN"] = field[3:].strip()
                    continue

                key, separator, value = field.partition("=")
                if not separator:
                    key, separator, value = field.partition(":")
                if not separator:
                    section["uncertainty"] = field
                    continue

                key = key.strip()
                value = value.strip()
                try:
                    parsed_value = float(value)
                    if "." not in value and "e" not in value.lower():
                        parsed_value = int(parsed_value)
                except ValueError:
                    parsed_value = value
                section[key] = parsed_value

    return out

def read_searm_radsys_1hz(fn="/Volumes/stu3data2/bnf_radsys_data/bnfradsys43mS10.00/bnfradsys43mS10.00.20260823.035959.raw.searm_radsys_1hz.00.20260823.030000.dat") -> xr.Dataset:
    with open(fn, newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        metadata = next(reader)
        columns = next(reader)
        units = next(reader)
        processing = next(reader)

    frame = pd.read_csv(
        fn,
        skiprows=4,
        header=None,
        names=columns,
        parse_dates=[0],
        date_format="%Y-%m-%d %H:%M:%S",
        engine="c",
    )
    timestamps = frame.pop(columns[0]).to_numpy(copy=False)
    data_vars = {
        name: (
            "time",
            frame[name].to_numpy(copy=False),
            {
                key: value
                for key, value in (
                    ("units", units[index]),
                    ("processing", processing[index]),
                )
                if value
            },
        )
        for index, name in enumerate(columns[1:], start=1)
    }
    ds = xr.Dataset(data_vars, coords={"time": timestamps})
    ds.attrs.update(
        zip(
            (
                "file_format",
                "station_name",
                "logger_model",
                "logger_serial_number",
                "logger_os",
                "program_name",
                "program_signature",
                "table_name",
            ),
            metadata,
        )
    )
    return ds

class A0:
    def __init__(self, config_file):
        self.config = read_config(config_file)

    
