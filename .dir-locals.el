((python-ts-mode
  . ((truncate-lines . 't)
     (eglot-workspace-configuration
      . ((:pylsp (plugins
                  ;(flake8 (enabled . :json-false))
                  (pycodestyle (enabled . :json-false))
                  (pyflakes (enabled . :json-false)))))))))
