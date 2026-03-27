/** Captures 'class_declaration' (Type). */
class MyJSClass {
    /** Captures 'field_definition' (Field). */
    myField = 1;

    /** Captures 'method_definition' (Method). */
    myMethod() {
        return this.myField;
    }
}

/** Captures 'function_declaration' (Method). */
function massiveStringFunction() {
    const massive = "This string is designed to be exceedingly long to force the chunker to split by size. " +
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]";
}

/** Captures 'generator_function_declaration' (Method). */
function* myGenerator() {
    yield 1;
}

/** Captures 'variable_declarator' with 'arrow_function' (Method). */
const myArrow = () => {
    return 2;
};

const myObjectLiteral = {
    /** Captures 'pair' with 'function_expression' (Accessor -> @method). */
    get myAccessor() {
        return 3;
    },
    myKey: function() {
        return 4;
    }
}
};
