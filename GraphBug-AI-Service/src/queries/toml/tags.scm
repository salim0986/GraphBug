; TOML Tags Query
; Captures tables/sections and key-value configuration pairs

; Top-level tables [table]
(table
  (bare_key) @definition.section)

(table
  (dotted_key
    (bare_key) @definition.section))

; Array of tables [[array]]
(table_array_element
  (bare_key) @definition.section)

(table_array_element
  (dotted_key
    (bare_key) @definition.section))

; Key-value pairs as configuration
(pair
  (bare_key) @definition.config)

(pair
  (quoted_key) @definition.config)

; Dotted keys (nested configuration)
(pair
  (dotted_key
    (bare_key) @definition.config))

; Inline table keys
(inline_table
  (pair
    (bare_key) @definition.config))
