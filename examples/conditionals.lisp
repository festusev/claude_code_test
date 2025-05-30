;; Conditional expressions
(if (> 5 3) "greater" "less or equal")

(if (= 2 2) 
    "equal" 
    "not equal")

;; Nested conditionals
(if (> 10 5)
    (if (< 3 7)
        "both true"
        "first true, second false")
    "first false")

;; Variables with conditionals
(define x 15)
(if (> x 10)
    (* x 2)
    (+ x 5))