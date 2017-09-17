-module(ann).
-export([
  perceptron/3, sigmoid/1, dot_prod/2, feed_forward/2,
  replace_input/2, convert_to_list/1,
  connect/2
]).

sigmoid(X) ->
  1 / (1 + math:exp(-X)).

dot_prod(X, Y) -> dot_prod(X, Y, 0).

dot_prod([], [], Acc) -> Acc;
dot_prod([X | Xs], [Y | Ys], Acc) ->
  dot_prod(Xs, Ys, Acc + X * Y).

feed_forward(Ws, Xs) ->
  sigmoid(dot_prod(Ws, Xs)).

perceptron(Ws, Inputs, Output_PIDs) ->
  receive
    {stimulate, Input} ->
      New_Inputs = replace_input(Inputs, Input),
      Y = feed_forward(Ws, convert_to_list(New_Inputs)),
      if Output_PIDs =/= [] ->
        lists:foreach(
          fun(Output_PID) ->
            Output_PID ! {stimulate, {self(), Y}}
          end,
          Output_PIDs
        );
        Output_PIDs =:= [] ->
          io:format("~p outputs: ~p~n", [self(), Y])
      end,
      perceptron(Ws, New_Inputs, Output_PIDs);
    {connect_to_output, Output_PID} ->
      New_Output_PIDs = [Output_PID | Output_PIDs],
      io:format("~p output connected to ~p: ~p~n", [self(), Output_PID, New_Output_PIDs]),
      perceptron(Ws, Inputs, New_Output_PIDs);
    {connect_to_input, Input_PID} ->
      New_Inputs = [{Input_PID, 0.5} | Inputs],
      io:format("~p inputs connected to ~p: ~p~n", [self(), Input_PID, New_Inputs]),
      perceptron(Ws, New_Inputs, Output_PIDs);
    {pass, X} ->
      Output = {self(), X},
      lists:foreach(
        fun(Output_PID) ->
          io:format("Stimulating ~p with ~p~n", [Output_PID, X]),
          Output_PID ! {stimulate, Output}
        end,
        Output_PIDs
      ),
      perceptron(Ws, Inputs, Output_PIDs)
  end.

connect(Input_PID, Output_PID) ->
  Input_PID ! {connect_to_output, Output_PID},
  Output_PID ! {connect_to_input, Input_PID}.

replace_input(Xs, Input) ->
  {Input_PID, _} = Input,
  lists:keyreplace(Input_PID, 1, Xs, Input).

convert_to_list(Tups) ->
  lists:map(
    fun(Tup) ->
      {_, X} = Tup,
      X
    end,
    Tups).

