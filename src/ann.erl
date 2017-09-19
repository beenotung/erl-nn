-module(ann).
-export([
  spawn/0,
  perceptron/3, sigmoid/1, dot_prod/2,
  replace_input/2,
  connect/2,
  add/2, mul/2, foldlZipWith/5
]).

-record(input, {pid, weight}).

-record(output, {pid, sensitivity}).

-record(backprop, {pid, sensitivity}).

%% export functions
spawn() ->
  spawn(ann, perceptron, [[], [], []]).

%% math

sigmoid(X) ->
  1 / (1 + math:exp(-X)).

sigmoid_deriv(X) ->
  math:exp(-X) / (1 + math:exp(-2 * X)).

add(X, Y) -> X + Y.

mul(X, Y) -> X * Y.

%%dot_prod(Xs, Ys) -> foldZipWith(fun add/2, fun mul/2, Xs, Ys, 0).

dot_prod(Xs, Ys) -> zipWithAcc(
  fun(Acc, X, Y) -> Acc + X * Y end,
  Xs,
  Ys,
  0
).

%% core logic

feed_forward(F, Ws, Xs) ->
  F(dot_prod(Ws, Xs)).

-spec perceptron(Weights, Inputs, Outputs) -> term() when
  Weights :: [integer()],
  Inputs :: [#input{}],
  Outputs :: [#output{}].

perceptron(Ws, Inputs, Outputs) ->
  Self = self(),
  receive
    {status, PID} ->
      PID ! {Ws, Inputs, Outputs},
      perceptron(Ws, Inputs, Outputs);
    {stimulate, Input} ->
      New_Inputs = replace_input(Inputs, Input),
      Y = feed_forward(fun sigmoid/1, Ws, convert_to_input_weights(New_Inputs)),
      if Outputs =/= [] ->
        Msg = {Self, Y},
        lists:foreach(
          fun(Output) ->
            Output#output.pid ! {stimulate, Msg}
          end,
          Outputs
        );
        Outputs =:= [] ->
          io:format("~p outputs: ~p~n", [Self, Y]),
          Self ! {learn, {Self, 1}}
      end,
      perceptron(Ws, New_Inputs, Outputs);
    {connect_to_output, Output_PID} ->
      New_Outputs = [#output{pid = Output_PID, sensitivity = 1} | Outputs],
      io:format("~p output connected to ~p: ~p~n", [Self, Output_PID, New_Outputs]),
      perceptron(Ws, Inputs, New_Outputs);
    {connect_to_input, Input_PID} ->
      W = 0.5,
      New_Inputs = [#input{pid = Input_PID, weight = W} | Inputs],
      io:format("~p inputs connected to ~p: ~p~n", [Self, Input_PID, New_Inputs]),
      perceptron([W | Ws], New_Inputs, Outputs);
    {pass, X} ->
      Msg = {Self, X},
      lists:foreach(
        fun(Output) ->
          io:format("Stimulating ~p with ~p~n", [Output#output.pid, X]),
          Output#output.pid ! {stimulate, Msg}
        end,
        Outputs
      ),
      perceptron(Ws, Inputs, Outputs);
    {learn, Backprop = #backprop{}} ->
      Learning_rate = 0.5,

      % Calculate the correct sensitivities
      New_Outputs = update_sensitivity(Outputs, Backprop),
      Xs = convert_to_input_weights(Inputs),
      Y = feed_forward(fun sigmoid/1, Ws, Xs),
      Deriv_Y = feed_forward(fun sigmoid_deriv/1, Ws, Xs),
      Sensitivity = calculate_sensitivity(Backprop, Inputs, New_Outputs, Y, Deriv_Y),

      io:format("(~p) New Sensitivities: ~p~n", [Self, New_Outputs]),
      io:format("(~p) Calculated Sensitivity: ~p~n", [Self, Sensitivity]),

      % Adjust all the weights
      New_Ws = lists:map(
        fun(Input) ->
          Learning_rate * Sensitivity * Input#input.weight
        end,
        Inputs
      ),
      io:format("(~p) Adjusted Weights: ~p~n", [Self, New_Ws]),

      % Propagate sensitivities and associated weights back to the previous layer
      lists:zipwith(
        fun(W, Input) ->
          Input#input.pid ! {learn, #input{pid = Self, weight = Sensitivity * W}}
        end,
        New_Ws,
        Inputs
      ),

      perceptron(New_Ws, Inputs, New_Outputs)
  end.

%% helper functions

connect(Input_PID, Output_PID) ->
  Input_PID ! {connect_to_output, Output_PID},
  Output_PID ! {connect_to_input, Input_PID}.

replace_input(Xs, Input) ->
  {Input_PID, _} = Input,
  lists:keyreplace(Input_PID, 1, Xs, Input).

convert_to_input_weights(Inputs) -> lists:map(fun(Input) ->
  Input#input.weight end, Inputs).

update_sensitivity(Outputs, #backprop{pid = Pid, sensitivity = S}) ->
  lists:map(
    fun(Output) ->
      if Output#output.pid == Pid -> Output#output{sensitivity = S};
        true -> Outputs
      end
    end,
    Outputs
  ).

calculate_sensitivity(Backprop, Inputs, Outputs, Y, Deriv_Y) ->
  if Inputs =:= [] ->
    % Input Node
    null;
    Outputs =:= [] ->
      % Output Node
      Training_Y = Backprop#input.weight,
      (Training_Y - Y) * Deriv_Y;
    true ->
      % Hidden Node
      Deriv_Y * lists:foldl(
        fun(Output, Acc) ->
          Output#output.sensitivity + Acc
        end,
        0,
        Outputs
      )
  end.

%% override std impl

zipWithAcc(_, [], [], Acc) -> Acc;
zipWithAcc(F, [X | Xs], [Y | Ys], Acc) ->
  zipWithAcc(F, Xs, Ys, F(Acc, X, Y)).

%% utils
foldlZipWith(F_Acc, F_Combine, Xs, Ys, Acc) ->
  lists:foldl(F_Acc, Acc, lists:zipwith(F_Combine, Xs, Ys)).
