from pyomo.core.base import ConcreteModel
from core.optimization import logger

class ConstraintFactory:
    factory_functions = {}

    @classmethod
    def add_to_model(cls, pyomo_model: ConcreteModel, key: str, specification: dict, **kwargs):
        """
        :param pyomo_model: The optimization model
        :param specification: Technical specification of the device model
        :param init_state: Initial state at the beginning of the optimization horizon
        """
        converter = cls.factory_functions[key]
        converter(pyomo_model, key, specification, **kwargs)  # calls factory function

    @classmethod
    def register_converter(cls, key, func):

        cls.factory_functions[key] = func
        logger.debug(f'Registered factory function for {key}.')
        logger.debug(f'All factory function registered so far: {cls.factory_functions}')
