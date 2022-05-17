(TeX-add-style-hook
 "bma"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "standalone"
    "standalone10"
    "amsmath"))
 :latex)

