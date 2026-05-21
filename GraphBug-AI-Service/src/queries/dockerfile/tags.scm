; Dockerfile Tags Query
; Captures instructions, environment variables, arguments, and build stages

; FROM instruction with optional stage name
(from_instruction
  (image_spec) @definition.config)

(from_instruction
  (image_spec)
  (image_alias) @definition.class)

; ENV declarations
(env_instruction
  (env_pair
    key: (unquoted_string) @definition.variable))

(env_instruction
  key: (unquoted_string) @definition.variable)

; ARG declarations
(arg_instruction
  (unquoted_string) @definition.variable)

(arg_instruction
  key: (unquoted_string) @definition.variable)

; LABEL instructions (metadata configuration)
(label_instruction
  (label_pair
    key: (unquoted_string) @definition.config))

; EXPOSE instructions (port configuration)
(expose_instruction
  (unquoted_string) @definition.config)

; WORKDIR instruction
(workdir_instruction
  (path) @definition.config)

; USER instruction
(user_instruction
  (unquoted_string) @definition.config)

; VOLUME instruction
(volume_instruction
  (path) @definition.config)

(volume_instruction
  (string_array
    (unquoted_string) @definition.config))

; ENTRYPOINT instruction
(entrypoint_instruction
  (shell_command) @definition.config)

(entrypoint_instruction
  (json_string_array) @definition.config)

; CMD instruction
(cmd_instruction
  (shell_command) @definition.config)

(cmd_instruction
  (json_string_array) @definition.config)

; HEALTHCHECK instruction
(healthcheck_instruction
  (cmd_instruction) @definition.config)

; STOPSIGNAL instruction
(stopsignal_instruction
  (unquoted_string) @definition.config)

; ONBUILD instruction
(onbuild_instruction) @definition.config
