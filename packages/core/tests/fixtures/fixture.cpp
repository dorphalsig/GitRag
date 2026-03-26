#include <string>

/**
 * Top-level namespace to capture 'namespace_definition' (Package).
 */
namespace MyCppNamespace {

/**
 * Top-level class to capture 'class_specifier' (Type).
 */
class MyCppClass {
public:
    /** Captures 'field_declaration' (Field) */
    int myClassField;
    
    /** Captures 'function_definition' (Method) for an inline class method */
    void doSomething() {
        myClassField++;
    }
};

/**
 * Top-level struct to capture 'struct_specifier' (Type).
 */
struct MyCppStruct {
    /** Captures 'field_declaration' (Field) */
    double myStructField;
};

/**
 * Top-level enum to capture 'enumerator' (EnumMember).
 */
enum MyCppEnum {
    /** Captures 'enumerator' */
    ENUM_ALPHA,
    /** Captures 'enumerator' */
    ENUM_BETA
};

/**
 * Captures 'function_definition' (Method) with a massive body to test length limits.
 */
void massiveStringFunction() {
    std::string massiveString = "This string is designed to be exceedingly long to force the chunker to split by size. "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]";
}

} // namespace MyCppNamespace
