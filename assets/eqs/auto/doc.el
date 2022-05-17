(TeX-add-style-hook
 "doc"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"))
 :latex)

