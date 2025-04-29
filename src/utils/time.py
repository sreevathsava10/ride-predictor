from datetime import datetime
from dateutil import parser
DATE_FMT = "%Y-%m-%d %H:%M:%S.%f %Z"


def iso_to_datetime(iso_str: str, date_format: str = DATE_FMT) -> datetime:
    """Converts a date in iso_str format to datetime object

    Args:
        iso_str: Date string in iso format, e.g. 2015-05-23 23:54:00.123 UTC
        date_format: Format of the date string to be parsed

    Returns:
        datetime object that represents the input date
    """
    return parser.parse(iso_str)


def hour_of_iso_date(iso_str: str, date_format: str = DATE_FMT) -> int:
    return iso_to_datetime(iso_str, date_format).hour


def robust_hour_of_iso_date(iso_date_str):
    if "UTC" not in iso_date_str:
        raise ValueError(f"Invalid ISO date string: {iso_date_str}")
    
    # Now continue your original parsing logic
    # Example (assuming datetime parsing):
    from datetime import datetime

    try:
        dt = datetime.strptime(iso_date_str, "%Y-%m-%d %H:%M:%S.%f UTC")
    except ValueError:
        dt = datetime.strptime(iso_date_str, "%Y-%m-%d %H:%M:%S UTC")
    
    return dt.hour



