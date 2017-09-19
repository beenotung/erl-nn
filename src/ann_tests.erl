-module(ann_tests).
-include_lib("eunit/include/eunit.hrl").

stimulate_test() ->
  Pid = spawn(ann, perceptron, [[0.5, 0.2], [{1, 0.6}, {2, 0.9}], [self()]]),
  Pid ! {stimulate, {1, 0.3}},
  receive Msg ->
    {stimulate, {_, X}} = Msg,
    _ = abs(X - 0.5817593768418363) < 0.000001,
    ok
  end.

connection_test() ->
  N1_PID = spawn(ann, perceptron, [[], [], []]),
  N2_PID = spawn(ann, perceptron, [[], [], []]),
  N3_PID = spawn(ann, perceptron, [[], [], []]),
  ann:connect(N1_PID, N2_PID),
  ann:connect(N1_PID, N3_PID),
  ok.

pass_test() ->
  N1_PID = spawn(ann, perceptron, [[], [], []]),
  N2_PID = spawn(ann, perceptron, [[], [], []]),
  N3_PID = spawn(ann, perceptron, [[], [], []]),
  ann:connect(N1_PID, N2_PID),
  ann:connect(N1_PID, N3_PID),
  N1_PID ! {pass, 0.5}.

run_test() ->
  X1_PID = ann:spawn(),
  X2_PID = ann:spawn(),

  H1_PID = ann:spawn(),
  H2_PID = ann:spawn(),

  O_PID = ann:spawn(),

  ann:connect(X1_PID, H1_PID),
  ann:connect(X1_PID, H2_PID),

  ann:connect(X2_PID, H1_PID),
  ann:connect(X2_PID, H2_PID),

  ann:connect(H1_PID, O_PID),
  ann:connect(H2_PID, O_PID),

  X1_PID ! {status, self()},
  X2_PID ! {status, self()},
  H1_PID ! {status, self()},
  H2_PID ! {status, self()},
  O_PID ! {status, self()},

  X1_PID ! {pass, 1.8},
  X2_PID ! {pass, 1.3}.
