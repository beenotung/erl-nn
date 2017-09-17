-module(ann).
-include_lib("eunit/include/eunit.hrl").
-compile(export_all).

feed_forward(Ws, Xs) ->
  maths:sigmoid(maths:dot_prod(Ws, Xs)).

feed_forward_deriv(Ws, Xs) ->
  maths:sigmoid_deriv(maths:dot_prod(Ws, Xs)).

perceptron(Weights, Inputs, Output_PIDs) ->
  receive
    {stimulate, Input} ->
      New_Inputs = replace_input(Inputs, Input),
      Output = feed_forward(Weights, convert_to_list(New_Inputs)),
      if
        Output_PIDs =/= [] ->
          lists:foreach(fun(Output_PID) -> Output_PID ! {stimulate, {self(), Output}} end, Output_PIDs);
        Output_PIDs == [] ->
          io:format("~p outputs: ~p~n", [self(), Output])
      end,
      perceptron(Weights, New_Inputs, Output_PIDs);

    {connect_to_output, Receiver_PID} ->
      New_Output_PIDs = [ Receiver_PID | Output_PIDs],
      io:format("~p output connected to ~p: ~p~n", [self(), Receiver_PID, New_Output_PIDs]),
      perceptron(Weights, Inputs, New_Output_PIDs);

    {connect_to_input, Sender_PID} ->
      New_Inputs = [{Sender_PID, 0.5} | Inputs],
      io:format("~p inputs connected to ~p: ~p~n", [self(), Sender_PID, New_Inputs]),
      perceptron([0.5|Weights], New_Inputs, Output_PIDs);

    {pass, Input_Value} ->
      lists:foreach(
        fun(Output_PID) ->
          io:format("~p stimulating ~p with ~p~n", [self(), Output_PID, Input_Value]),
          Output_PID ! {stimulate, {self(), Input_Value}}
        end,
        Output_PIDs),
      perceptron(Weights, Inputs, Output_PIDs)

  end.

connect(Sender_PID, Receiver_PID) ->
  Receiver_PID ! {connect_to_input, Sender_PID},
  Sender_PID ! {connect_to_output, Receiver_PID}.

replace_input(Inputs, Input) ->
  {Input_PID, _} = Input,
  lists:keyreplace(Input_PID, 1, Inputs, Input).

convert_to_list(Inputs) ->
  lists:map(fun(Tup) -> {_, X} = Tup, X end, Inputs).


