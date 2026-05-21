; HCL (Terraform) Tags Query
; Captures resources, data sources, variables, outputs, and modules

; Resource blocks: resource "type" "name" { }
(block
  (identifier) @_type
  (#eq? @_type "resource")
  (string_lit) @_resource_type
  (string_lit) @definition.resource)

; Data source blocks: data "type" "name" { }
(block
  (identifier) @_type
  (#eq? @_type "data")
  (string_lit) @_data_type
  (string_lit) @definition.resource)

; Variable declarations: variable "name" { }
(block
  (identifier) @_type
  (#eq? @_type "variable")
  (string_lit) @definition.variable)

; Output declarations: output "name" { }
(block
  (identifier) @_type
  (#eq? @_type "output")
  (string_lit) @definition.variable)

; Local values: locals { name = value }
(block
  (identifier) @_type
  (#eq? @_type "locals")
  (body
    (attribute
      (identifier) @definition.variable)))

; Module blocks: module "name" { }
(block
  (identifier) @_type
  (#eq? @_type "module")
  (string_lit) @definition.class)

; Provider blocks: provider "name" { }
(block
  (identifier) @_type
  (#eq? @_type "provider")
  (string_lit) @definition.class)

; Terraform blocks: terraform { }
(block
  (identifier) @_type
  (#eq? @_type "terraform")
  @definition.class)

; Top-level attributes (configuration)
(attribute
  (identifier) @definition.variable)
