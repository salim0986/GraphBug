; HTML Tags Query
; Captures HTML elements, attributes, and structural components

; HTML elements with id attribute (important structural identifiers)
(element
  (start_tag
    (attribute
      (attribute_name) @_attr_name
      (#eq? @_attr_name "id")
      (quoted_attribute_value
        (attribute_value) @definition.variable))))

; HTML elements with class attribute (styling/component identifiers)
(element
  (start_tag
    (attribute
      (attribute_name) @_attr_name
      (#eq? @_attr_name "class")
      (quoted_attribute_value
        (attribute_value) @definition.config))))

; Form elements with name attribute (form field identifiers)
(element
  (start_tag
    (tag_name) @_tag
    (#match? @_tag "^(input|select|textarea|button|form)$")
    (attribute
      (attribute_name) @_attr_name
      (#eq? @_attr_name "name")
      (quoted_attribute_value
        (attribute_value) @definition.variable))))

; Script tags (embedded JavaScript)
(script_element
  (start_tag
    (tag_name) @definition.config))

; Style tags (embedded CSS)
(style_element
  (start_tag
    (tag_name) @definition.config))

; Custom elements / Web Components (contain hyphen)
(element
  (start_tag
    (tag_name) @definition.class
    (#match? @definition.class "^[a-z]+-[a-z-]+$")))

; Data attributes (data-* attributes for custom data)
(element
  (start_tag
    (attribute
      (attribute_name) @definition.config
      (#match? @definition.config "^data-"))))

; Template elements (HTML templates)
(element
  (start_tag
    (tag_name) @_tag
    (#eq? @_tag "template")
    (attribute
      (attribute_name) @_attr_name
      (#eq? @_attr_name "id")
      (quoted_attribute_value
        (attribute_value) @definition.class))))

; Semantic section elements
(element
  (start_tag
    (tag_name) @definition.section
    (#match? @definition.section "^(section|article|aside|nav|main|header|footer)$")))
