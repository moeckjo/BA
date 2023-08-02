'''
- Determines momentary flex for each device
- Derives flex at grid-connection point (for grid operator)

Open questions:
-> which format? max & min feed-in & consumption, respectively?
'''
import os

import typing
from devicemanagement import logger
from devicemanagement.utils import get_latest_measurement
from devicemanagement.models.device_model import Device


def electric_flexibilities(devices: typing.Dict[str, Device], t_delta: float) -> typing.Dict[str, typing.List[int]]:
    """
    Get max. and min. consumption and generation power for all electric devices, respectively.
    :param devices: Dictionary containing the device model corresponding to its key
    :param t_delta: Length of the period for which these power values shall be maintained
    :return: Dictionary with:
            - max_generation: List of upper limits of generation devices and battery discharge
            - min_generation:  List of lower limits of generation devices and battery discharge
            - max_load: List of upper consumption limits of flexible devices, battery charge and the given inflexible load
            - min_load: List of lower consumption limits of flexible devices, battery charge and the given inflexible load

    """
    def append(value: typing.Union[float, None], to_list: list, value_name: str):
        """
        Handle None value when trying to convert it to int before appending it to the corresponding list.
        :param value: Value to append.
        :param to_list: List to the append the value to.
        :param value_name: Name of the value/variable, e.g. "max_gen", used in the logging message.
        :return:
        """
        try:
            to_list.append(int(value))
        except ValueError as e:
            if "cannot convert float NaN to integer" in str(e):
                logger.warning(f"Value for '{value_name}' of device {key} is {value}. "
                               f"Cannot convert float NaN to integer. Appending value as-is to the list.")
                to_list.append(value)
            else:
                raise e

    # TODO: Does HP react fast enough for red phase?

    # Inflexible sources
    load = get_latest_measurement(source=os.getenv('LOAD_EL_KEY'), fields='active_power')

    max_generation = []
    min_generation = []
    max_load = [load]
    min_load = [load]

    for key, model in devices.items():
        if model.subcategory == os.getenv('HEAT_PUMP_KEY'):
            # Only get the flexibility of heat pumps if they are controllable, because otherwise they are part
            # of the inflexible load
            if model.controllable:
                max_gen, min_gen, max_cons, min_cons = model.flexibility(
                    t_delta, heat_storage=devices[os.getenv('HEAT_STORAGE_KEY')]
                )
            else:
                continue
        elif model.subcategory == os.getenv('HEAT_STORAGE_KEY'):
            continue
        elif model.subcategory == os.getenv('PV_KEY'):
            flex = model.flexibility(t_delta)  # list of output levels based on curtailment levels
            max_gen, min_gen, max_cons, min_cons = (min(flex), max(flex), 0, 0)
        else:
            # All other cases. Throws AttributeError, when model has no flexibility method or TypeError
            # if flexibility method returns nothing/None
            try:
                max_gen, min_gen, max_cons, min_cons = model.flexibility(t_delta)
            except AttributeError:
                logger.warning(f"Device {key} (of class {type(model)}) has no flexibility method. "
                               f"It should be implemented.")
                continue
            except TypeError:
                continue

        append(value=max_gen, to_list=max_generation, value_name="max_gen")
        append(value=min_gen, to_list=min_generation, value_name="min_gen")
        append(value=max_cons, to_list=max_load, value_name="max_cons")
        append(value=min_cons, to_list=min_load, value_name="min_cons")

    return dict(max_generation=max_generation, min_generation=min_generation, max_load=max_load, min_load=min_load)
