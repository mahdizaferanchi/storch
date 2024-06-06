configuration = {}

def use_config(config):
    global configuration
    configuration = config

def configurable(func):
    """Decorator that makes all functions it wraps configurable.
    The configuration object should be passed to `use_config` 
    """
    import inspect
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract the namespace (if provided)
        namespace = kwargs.pop('_c', None)
        
        # Get the function name
        func_name = func.__name__
        
        # Get the function signature
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs).arguments

        # Determine the search order for the config values
        config_sources = []

        if namespace:
            if namespace in configuration:
                config_sources.append(configuration[namespace])
            if func_name in configuration and namespace in configuration[func_name]:
                config_sources.append(configuration[func_name][namespace])

        config_sources.append(configuration)

        # Update the bound arguments with values from the config sources
        for name, param in sig.parameters.items():
            if name not in bound_args:
                for config in config_sources:
                    if name in config:
                        bound_args[name] = config[name]
                        break

        # Call the function with the resolved arguments
        return func(**bound_args)

    return wrapper