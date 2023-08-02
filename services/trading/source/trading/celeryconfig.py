# Task settings
task_serializer = 'pickle'
accept_content = ['json', 'pickle']
task_routes = {
    'core.*': {'queue': 'core'},
    'trading.*': {'queue': 'trading'},
    'gridmgmt.*': {'queue': 'gridmgmt'},

}
