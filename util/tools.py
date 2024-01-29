

    
def get_instance(module, class_name, *args, **kwargs):
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

