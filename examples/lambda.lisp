;; Lambda expressions
(lambda (x) (* x x))

;; Lambda with multiple parameters
(lambda (x y) (+ x y))

;; Using lambda in function calls
((lambda (x) (* x 2)) 5)

;; Lambda with conditionals
((lambda (x) 
   (if (> x 0) 
       "positive" 
       "non-positive")) -3)

;; Higher-order function example
(define apply-twice (lambda (f x) (f (f x))))
(define increment (lambda (x) (+ x 1)))
(apply-twice increment 5)