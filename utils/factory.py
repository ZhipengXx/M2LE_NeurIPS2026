def get_model(model_name, args):
    name = model_name.lower()
    if name == 'm2le':
        from methods.m2le import Learner
        return Learner(args)
    raise ValueError(f"Unknown model '{model_name}'. Expected 'm2le'.")
