; Bash/Shell Script Tags Query
; Captures functions, variables, and shell constructs

; Function definitions
(function_definition
  name: (word) @definition.function)

; Variables assignments
(variable_assignment
  name: (variable_name) @definition.variable)

; Command substitutions that define variables
(command
  name: (command_name) @_cmd
  (#match? @_cmd "^(local|declare|export|readonly)$")
  argument: (word) @definition.variable)

; For loop variables
(for_statement
  variable: (variable_name) @definition.variable)

; Select statement variables
(select_statement
  name: (variable_name) @definition.variable)

; Case patterns (useful for understanding control flow)
(case_statement
  value: (word) @definition.variable)

; Exported variables/functions
(declaration_command
  (variable_assignment
    name: (variable_name) @definition.variable))
