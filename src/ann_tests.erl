-module(ann_tests).
-include_lib("eunit/include/eunit.hrl").

stimulate_test() ->
  Pid = spawn(ann, perceptron, [[0.5, 0.2], [{1, 0.6}, {2, 0.9}], [self()]]),
  Pid ! {stimulate, {1, 0.3}},
  receive Msg ->
    {stimulate, {_, X}} = Msg,
    abs(X - 0.5817593768418363) < 0.000001,
    ok
  end.

connection_test() ->
  N1_PID = spawn(ann, perceptron, [[],[],[]]),
  N2_PID = spawn(ann, perceptron, [[],[],[]]),
  N3_PID = spawn(ann, perceptron, [[],[],[]]),
  ann:connect(N1_PID, N2_PID),
  ann:connect(N1_PID, N3_PID),
  ok.

pass_test() ->
  N1_PID = spawn(ann, perceptron, [[],[],[]]),
  N2_PID = spawn(ann, perceptron, [[],[],[]]),
  N3_PID = spawn(ann, perceptron, [[],[],[]]),
  ann:connect(N1_PID, N2_PID),
  ann:connect(N1_PID, N3_PID),
  N1_PID ! {pass, 0.5}.
