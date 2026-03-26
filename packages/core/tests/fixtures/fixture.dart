/// Captures 'library_name' (Package).
library my_dart_library;

/// Captures 'mixin_declaration' (Type).
mixin MyDartMixin {
  /// Captures 'type_identifier' within declaration (Field).
  int mixinField = 0;
}

enum MyDartEnum {
  /// Captures 'enum_constant' (EnumMember).
  alpha,
  /// Captures 'enum_constant' (EnumMember).
  beta
}

/// Captures 'class_definition' (Type).
class MyDartClass with MyDartMixin {
  /// Captures 'type_identifier' within declaration (Field).
  String myStringField;

  /// Captures 'getter_signature' (Accessor).
  String get myProperty => myStringField;

  /// Captures 'setter_signature' (Accessor).
  set myProperty(String value) {
    myStringField = value;
  }

  /// Captures 'constructor_signature' (Constructor).
  MyDartClass(this.myStringField);

  /// Captures 'method_signature' (Method).
  void doSomething() {
    print(myStringField);
  }

  /// Captures 'method_signature' (Method) to test length limits.
  void massiveStringMethod() {
    String massive = "This string is designed to be exceedingly long to force the chunker to split by size. "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] "
    "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]";
  }
}
