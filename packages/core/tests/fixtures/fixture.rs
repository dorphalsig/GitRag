// Captures 'attribute_item' (AnnotationElement).
#[derive(Debug)]
// Captures 'struct_item' (Type).
struct MyRustStruct {
    // Captures 'field_declaration' (Field).
    my_field: i32,
}

// Captures 'trait_item' (Type).
trait MyRustTrait {
    fn do_something(&self);
}

// Captures 'enum_item' (Type).
enum MyRustEnum {
    // Captures 'enum_variant' (EnumMember).
    VariantOne,
    // Captures 'enum_variant' (EnumMember).
    VariantTwo,
}

// Captures 'impl_item' containing methods (Method grouping).
impl MyRustStruct {
    // Captures 'function_item' inside impl_item (Method).
    fn new() -> Self {
        MyRustStruct { my_field: 0 }
    }
}

// Captures 'function_item' at the top level (Method).
fn massive_string_function() {
    let massive = "This string is designed to be exceedingly long to force the chunker to split by size. \
    Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] \
    Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] \
    Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] \
    Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] \
    Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] \
    Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] \
    Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] \
    Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]";
}
