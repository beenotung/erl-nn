%
% Direct Feedback Alignment Neural Network
%
% Training without Backpropagation
%
% Reference:
% https://medium.com/@rilut/neural-networks-without-backpropagation-direct-feedback-alignment-30d5d4848f5
%
-module(dfnn).
-include_lib("eunit/include/eunit.hrl").
-compile(export_all).

-record(node, {
  bias :: number(),
  inputs = [] :: [{pid(), weight()}],
  outputs = [] :: [pid()]
}).

-type weight() :: number().

sigmoid(X) ->
  1 / (1 + math:exp(-X)).

df_sigmoid(X) ->
  sigmoid(X) * (1 - sigmoid(X)).

-spec init(integer(), [integer()], integer()) -> todo.
init(N_Input, N_Hidden_List, N_Output) ->
  Ns = lists:flatten([N_Input, N_Hidden_List, N_Output]),
  Layers = lists:map(fun create_layer/1, Ns),
  lists:foldl(fun(Output_Layer, Input_Layer) ->
    connect_layer(Input_Layer, Output_Layer),
    Output_Layer
              end, [], Layers),
  todo.

create_layer(N) ->
  create_layer(N, []).

create_layer(0, Acc) when is_list(Acc) -> Acc;
create_layer(N, Acc) when N > 0 ->
  Pid = spawn(fun() -> loop_node(#node{bias = random:uniform()}) end),
  create_layer(N - 1, [Pid | Acc]).

-spec connect_layer(pid(), pid()) -> ok.
connect_layer(Inputs, Outputs) ->
  lists:foreach(fun(Output_Pid) ->
    lists:foreach(fun(Input_Pid) ->
      Output_Pid ! {connect, {Input_Pid, random:uniform()}}
                  end, Inputs)
                end, Outputs).

loop_node(Node = #node{}) ->
  receive
    {connect, Input = {Input_Pid, Weight}} when is_pid(Input_Pid), is_number(Weight) ->
      io:fwrite("[log] ~p connect to ~p with weight ~p~n", [Input_Pid, self(), Weight]),
      New_Inputs = [Input | Node#node.inputs],
      loop_node(Node#node{inputs = New_Inputs});
    X ->
      io:fwrite("[debug] ~p received ~p~n.", [self(), X]),
      loop_node(Node)
  end.

