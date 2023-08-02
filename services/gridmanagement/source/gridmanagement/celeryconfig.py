# Task settings
task_serializer = 'pickle'
accept_content = ['json', 'pickle']
task_routes = {
    'core.*': {'queue': 'core'},
    'devmgmt.*': {'queue': 'devmgmt'},
    'gridmgmt.*': {'queue': 'gridmgmt'},
    'trading.*': {'queue': 'trading'}
}
worker_send_task_events = True
