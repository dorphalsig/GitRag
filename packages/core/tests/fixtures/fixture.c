#include <stdio.h>

/**
 * Top-level struct to capture 'struct_specifier' as a Type.
 */
struct MyCStruct {
    /** Captures 'field_declaration' inside the struct */
    int myField;
    /** Captures a second 'field_declaration' */
    char myChar;
};

/**
 * Standard function to capture 'function_definition' as a Method.
 * This also tests the size-splitting fallback by exceeding HARD_CAP_BYTES
 * with a single massive string literal that contains no child statements.
 */
void massiveStringFunction() {
    const char* massiveString = "This string is designed to be exceedingly long to force the chunker to split by size. "
    "Because this string literal has no internal statement nodes, Tree-sitter's _split_large_node will gracefully "
    "fall back to _newline_aligned_ranges and slice this across multiple chunks, nudging the cuts to newlines. "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]";
}

/**
 * Another 'function_definition' representing standard executable code.
 */
int main() {
    struct MyCStruct s;
    s.myField = 1;
    return 0;
}
