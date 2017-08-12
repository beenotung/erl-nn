-module(maths).
-compile(export_all).

multiple(A, B) -> A * B.

dot_prod(Xs, Ys) -> lists:zipwith(fun multiple/2, Xs, Ys).

sigmoid(X) -> 1 / (1 + math:exp(-X)).
