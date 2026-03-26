<?php

/** Captures 'namespace_definition' (Package). */
namespace MyPhpNamespace;

/** Captures 'interface_declaration' (Type). */
interface MyPhpInterface {
    public function doSomething();
}

/** Captures 'trait_declaration' (Type). */
trait MyPhpTrait {
    /** Captures 'property_declaration' (Field). */
    public $traitProperty = 1;
}

/** Captures 'class_declaration' (Type). */
class MyPhpClass implements MyPhpInterface {
    use MyPhpTrait;

    /** Captures 'property_declaration' (Field). */
    private $myField;

    /** Captures 'method_declaration' (Method). */
    public function __construct() {
        $this->myField = 0;
    }

    /** Captures 'method_declaration' (Method). */
    public function doSomething() {
        $this->myField++;
    }

    /** Captures 'method_declaration' (Method) testing HARD_CAP_BYTES. */
    public function massiveStringMethod() {
        $massive = "This string is designed to be exceedingly long to force the chunker to split by size. " .
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " .
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " .
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " .
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " .
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " .
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " .
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX] " .
        "Padding: [XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX]";
    }
}

/** Captures 'function_definition' (Method) at the top-level. */
function myTopLevelFunction() {
    return true;
}
