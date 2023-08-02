import datetime
import os

import utils
from forecasting.load_forecast_service import ReferenceBasedLoadForecaster
from forecasting.ev_forecast_service import ReferenceBasedEVForecaster
from forecasting.pv_forecast_service import ShortTermForecaster, QuotaBlocksForecaster

'''
Only for development and testing of new forecast models
'''


def make_ev_forecast(window_start):
    # last_full_hour = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
    # # Predict load as of next full hour for the following 24 hours -> adjust to your needs
    # window_start = last_full_hour + datetime.timedelta(hours=1)
    window_size = datetime.timedelta(hours=24)
    forecaster = ReferenceBasedEVForecaster(window_start, window_size)
    forecast = forecaster.make_forecast()
    print(f'Forecast of {forecaster.source} for next {window_size} hour(s): {forecast}')  #: {load_forecast}.')


def make_load_forecast(window_start):
    # last_full_hour = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
    # # Predict load as of next full hour for the following 24 hours -> adjust to your needs
    # window_start = last_full_hour + datetime.timedelta(hours=1)
    window_size = datetime.timedelta(hours=24)
    load_forecaster = ReferenceBasedLoadForecaster(window_start, window_size)
    load_forecast = load_forecaster.make_forecast()
    print(f'Forecast of el. load for next {window_size} hour(s): {load_forecast}')  #: {load_forecast}.')


def make_pv_forecast(window_start, which='long'):
    # Predict pv as of next full hour for the following 24 hours -> adjust to your needs
    # last_full_hour = datetime.datetime.now(tz=datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
    # # window_start = last_full_hour + datetime.timedelta(hours=3)
    # which = 'long' # 'short'
    # Only MOSMIX data available for testing 2021-04-20 - 2021-04-30 -> use a preceding day
    # pv_specifications: dict = utils.get_devices_of_subcategory(os.getenv('PV_KEY'))[0]
    # pv_specifications['key'] = 'pv'
    pv_specifications = {'key': 'pv', 'category': 'generator', 'subcategory': 'pv', 'energy_carrier': 'el',
                         'active_power_nominal': -10000,
                         'inverter_efficiency': 0.97, 'continuous_curtailment': False,
                         'curtailment_levels': [0, 0.3, 0.6, 1]}

    print(f'PV specs: {pv_specifications}')

    window_start = window_start  # datetime.datetime(2021,4,20,10,0, tzinfo=datetime.timezone.utc)
    if which == 'short':
        window_size = datetime.timedelta(hours=1)
        pv_forecaster = ShortTermForecaster(pv_specification=pv_specifications, window_start=window_start, window_size=window_size)
    else:
        window_size = datetime.timedelta(hours=24)
        pv_forecaster = QuotaBlocksForecaster(window_start=window_start, window_size=window_size,
                                              pv_specification=pv_specifications)
    pv_forecast = pv_forecaster.make_forecast()
    print(f'PV Forecast for next {window_size} hour(s): {pv_forecast}')  #: {load_forecast}.')


if __name__ == '__main__':
    window_start = datetime.datetime(2022, 1, 1, 9, 0, tzinfo=datetime.timezone.utc)
    lead_time = datetime.timedelta(seconds=int(os.getenv('PV_FORECAST_SHORTTERM_LEAD_TIME_SEC')))
    # window_start = (datetime.datetime.now(tz=datetime.timezone.utc) + lead_time).replace(second=0, microsecond=0)
    # make_load_forecast(window_start)
    make_pv_forecast(window_start, "long")
    # make_ev_forecast(window_start)
