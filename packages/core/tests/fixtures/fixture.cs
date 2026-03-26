using System;

/**
 * Captures 'namespace_declaration' (Package).
 */
namespace MyCSharpNamespace {

/**
 * Custom attribute definition.
 */
public class MyCustomAttribute : Attribute {}

/**
 * Top-level interface to capture 'interface_declaration' (Type).
 */
public interface IMyInterface {
    void Execute();
}

/**
 * Top-level struct to capture 'struct_declaration' (Type).
 */
public struct MyCSharpStruct {
    /** Captures 'field_declaration' (Field) */
    public int structField;
}

/**
 * Top-level record to capture 'record_declaration' (Type).
 */
public record MyCSharpRecord(int Id);

/**
 * Top-level enum to capture 'enum_member_declaration' (EnumMember).
 */
public enum MyCSharpEnum {
    /** Captures 'enum_member_declaration' */
    StateOne,
    /** Captures 'enum_member_declaration' */
    StateTwo
}

/**
 * Captures 'attribute' (AnnotationElement) attached to a class.
 * Captures 'class_declaration' (Type).
 */
[MyCustomAttribute]
public class MyCSharpClass : IMyInterface {
    /** Captures 'field_declaration' (Field) */
    private string myStringField;

    /** Captures 'property_declaration' (Accessor) */
    public int MyProperty { get; set; }

    /** Captures 'indexer_declaration' (Accessor) */
    public int this[int index] {
        get { return index; }
        set { }
    }

    /** Captures 'constructor_declaration' (Constructor) */
    public MyCSharpClass() {
        myStringField = "initialized";
    }

    /** Captures 'method_declaration' (Method) */
    public void Execute() {
        // ...
    }

    /**
     * Captures 'method_declaration' with a massive string assignment to test limits.
     */
    public void MassiveStringMethod() {
        string massive = "This string is designed to be exceedingly long to force the chunker to split by size. " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " +
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]";
    }
}

}
