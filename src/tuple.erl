-module(tuple).
-compile(export_all).

-spec last(dict:dict()) -> any().
last({_, X}) -> X.

-spec list_last(list()) -> any().
list_last(Xs) -> lists:map(fun last/1, Xs).
