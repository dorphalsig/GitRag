"""Module-level docstring."""

def my_decorator(func):
    """Captures 'function_definition' (Method)."""
    return func

# Captures 'decorator' (AnnotationElement)
@my_decorator
class MyPythonClass:
    """Captures 'class_definition' (Type)."""
    
    # Captures 'expression_statement (assignment)' (Field)
    my_field = 42
    
    def __init__(self):
        """Captures 'function_definition' named '__init__' (Initializer)."""
        self.value = 1

    # Captures 'decorator' with name 'property' to trigger (Accessor)
    @property
    def my_property(self):
        """Captures 'decorated_definition' (Accessor)."""
        return self.value

    def massive_string_function(self):
        """Captures 'function_definition' (Method) testing HARD_CAP_BYTES fallback."""
        massive_string = (
            "This string is designed to be exceedingly long to force the chunker to split by size. "
            "Because this string literal has no internal statement nodes, Tree-sitter's fallback kicks in. "
            "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
            "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
            "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
            "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
            "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
            "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
            "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
            "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]"
        )
