# Task settings
task_serializer = 'pickle'
task_routes = {
    'core.*': {'queue': 'core'},
    'devmgmt.*': {'queue': 'devmgmt'},
    'gridmgmt.*': {'queue': 'gridmgmt'},
    'trading.*': {'queue': 'trading'}

}

