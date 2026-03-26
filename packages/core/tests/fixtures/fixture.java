/** Captures 'package_declaration' (Package) */
package com.example.myjava;

/**
 * Top-level annotation to capture 'annotation_type_element_declaration' (AnnotationElement).
 */
@interface MyJavaAnnotation {
    /** Captures 'annotation_type_element_declaration' */
    String value();
}

/**
 * Top-level interface to capture 'interface_declaration' (Type).
 */
interface MyJavaInterface {
    void process();
}

/**
 * Top-level enum to capture 'enum_declaration' (Type) and 'enum_constant' (EnumMember).
 */
enum MyJavaEnum {
    /** Captures 'enum_constant' */
    ALPHA,
    /** Captures 'enum_constant' */
    BETA;
}

/**
 * Top-level record to capture 'record_declaration' (Type).
 */
record MyJavaRecord(
    /** Captures 'formal_parameter' (RecordComponent) */
    int componentId
) {}

/**
 * Top-level class to capture 'class_declaration' (Type).
 */
public class MyJavaClass implements MyJavaInterface {
    /** Captures 'field_declaration' (Field) */
    private int myField;

    /** Captures 'static_initializer' (Initializer) */
    static {
        System.out.println("Static initialization");
    }

    /** Captures 'class_body (block)' (Initializer) representing an instance init block. */
    {
        myField = 100;
    }

    /** Captures 'constructor_declaration' (Constructor) */
    public MyJavaClass() {
        myField = 0;
    }

    /** Captures 'method_declaration' (Method) */
    @Override
    public void process() {
        // Implementation
    }

    /**
     * Captures 'method_declaration' with a massive string for the chunker cap testing.
     */
    public void massiveStringMethod() {
        String massive = "This string is designed to be exceedingly long to force the chunker to split by size. " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]";
    }
}
