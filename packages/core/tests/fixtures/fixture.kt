/** Captures 'package_header' (Package) */
package com.example.mykotlin

/**
 * Top-level object to capture 'object_declaration' (Type).
 */
object MyKotlinObject {
    /** Captures 'property_declaration' (Field) */
    val objectProperty: Int = 1
}

/**
 * Top-level enum class to capture 'class_declaration' (Type) and 'enum_entry' (EnumMember).
 */
enum class MyKotlinEnum {
    /** Captures 'enum_entry' */
    ENTRY_ONE,
    /** Captures 'enum_entry' */
    ENTRY_TWO
}

/**
 * Top-level class to capture 'class_declaration' (Type).
 */
class MyKotlinClass {
    /** Captures 'property_declaration' (Field). */
    var customProperty: String = "default"
        /** Captures 'getter' (Accessor) */
        get() = field
        /** Captures 'setter' (Accessor) */
        set(value) {
            field = value
        }

    /** Captures 'anonymous_initializer' (Initializer) */
    init {
        println("Instance initialization")
    }

    /** Captures 'secondary_constructor' (Method, per your config) */
    constructor(overrideValue: String) {
        customProperty = overrideValue
    }

    /** Captures 'function_declaration' (Method) with a massive string assignment. */
    fun massiveStringFunction() {
        val massive = """
            This string is designed to be exceedingly long to force the chunker to split by size.
            Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]
            Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]
            Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]
            Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]
            Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]
            Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]
            Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]
            Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]
        """.trimIndent()
    }
}

/**
 * Captures 'function_declaration' (Method) at the top level.
 */
fun myTopLevelFunction() {
    println("Executing top level")
}
