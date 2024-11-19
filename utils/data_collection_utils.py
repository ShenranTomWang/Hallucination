import importlib

def generate(model, tokenizer, query, device):
    input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids, max_length=10000)[0, len(input_ids):]
    answer = tokenizer.decode(output_ids)
        
    return answer

def get_class(module_name, class_name):
    """
    Dynamically loads a class by its module and class name.
    Args:
        module_name (str): The module's name (e.g., 'my_classes').
        class_name (str): The class name to retrieve (e.g., 'MyClass').
    Returns:
        class: The class object.
    """
    # Dynamically import the module
    module = importlib.import_module(module_name)
    
    # Retrieve the class from the module
    cls = getattr(module, class_name)
    
    return cls
    