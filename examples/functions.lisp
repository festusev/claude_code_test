;; Function definitions
(defun square (x)
  (* x x))

(defun add-two (a b)
  (+ a b))

(defun factorial (n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))

;; Function calls
(square 5)
(add-two 3 7)
(factorial 5)