; Lua Tags Query
; Captures functions, variables, tables, and module definitions

; Function definitions (global and local)
(function_declaration
  name: (identifier) @definition.function)

(function_declaration
  name: (dot_index_expression
    field: (identifier) @definition.function))

(function_declaration
  name: (method_index_expression
    method: (identifier) @definition.function))

; Local function definitions
(local_function_declaration
  name: (identifier) @definition.function)

; Assignment to functions
(assignment_statement
  (variable_list
    name: (identifier) @definition.function)
  (expression_list
    value: (function_definition)))

; Variable assignments (global)
(assignment_statement
  (variable_list
    name: (identifier) @definition.variable))

; Local variable declarations
(local_variable_declaration
  (variable_list
    name: (identifier) @definition.variable))

; For loop variables
(for_numeric_statement
  name: (identifier) @definition.variable)

(for_generic_statement
  (variable_list
    name: (identifier) @definition.variable))

; Table fields as properties
(field
  name: (identifier) @definition.variable)

; Module patterns - require and module definitions
(assignment_statement
  (variable_list
    name: (identifier) @definition.class)
  (expression_list
    value: (table_constructor)))

; Module returns (common Lua module pattern)
(return_statement
  (expression_list
    (table_constructor) @definition.class))
