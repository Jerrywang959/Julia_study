sysuse auto.dta
br
reg price rep78 mpg
test rep78==0
test rep78==mpg
