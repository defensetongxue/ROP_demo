from . import meta_model
def build_model(configs):
    
    model= getattr(meta_model,f"build_{configs['name']}")(configs)
    return model