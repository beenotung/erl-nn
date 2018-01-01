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
  % pid() -> weight()
  input_weights = maps:new() :: map(),
  % pid() -> weight()
  input_values = maps:new() :: map(),
  outputs = [] :: [pid()],
  is_output = false :: boolean()
}).

-type weight() :: number().
-type network() :: [[pid()]].
-type train_data() :: {[number()], [number()]}.

sigmoid(X) ->
  1 / (1 + math:exp(-X)).

df_sigmoid(X) ->
  sigmoid(X) * (1 - sigmoid(X)).

-spec init(integer(), [integer()], integer()) -> network().
init(N_Input, N_Hidden_List, N_Output) ->
  Ns = lists:flatten([N_Input, N_Hidden_List, N_Output]),
  Layers = lists:map(fun create_layer/1, Ns),
  Output_Layer = lists:foldl(
    fun(Output_Layer, Input_Layer) ->
      connect_layer(Input_Layer, Output_Layer),
      Output_Layer
    end, [], Layers),
  lists:map(fun(Output_Pid) -> Output_Pid ! mark_output end, Output_Layer),
  Layers.

create_layer(N) ->
  create_layer(N, []).

create_layer(0, Acc) when is_list(Acc) -> Acc;
create_layer(N, Acc) when N > 0 ->
  Pid = spawn(fun() -> loop_node(#node{bias = rand:uniform()}) end),
  create_layer(N - 1, [Pid | Acc]).

-spec connect_layer(pid(), pid()) -> ok.
connect_layer(Inputs, Outputs) ->
  lists:foreach(fun(Output_Pid) ->
    lists:foreach(fun(Input_Pid) ->
      Weight = rand:uniform(),
      Output_Pid ! {connect, {Input_Pid, Weight}}
                  end, Inputs)
                end, Outputs).

loop_node(Node = #node{}) ->
  receive
    {input, {Input, Input_Pid}, Report_Pid} when is_number(Input), is_pid(Input_Pid), is_pid(Report_Pid) ->
      N_Input = maps:size(Node#node.input_weights),
      X =
        if
          N_Input == 0 ->
            % input layer, no weighting
            Input;
          true ->
            % hidden layer or output layer, to use weight to adjust the input
            Weight = maps:get(Input_Pid, Node#node.input_weights),
            Input * Weight
        end,
      Input_Values1 = maps:put(Input_Pid, X, Node#node.input_values),
      N_Xs = maps:size(Input_Values1),
      Input_Values2 =
        if
          N_Input == N_Xs ->
            % send forward
            Sum = lists:foldl(fun add/2, 0, maps:values(Input_Values1)),
            Output = sigmoid(Sum),
            if
              length(Node#node.outputs) == 0 ->
                % output layer
                Report_Pid ! {self(), Output};
              true ->
                % input layer or hidden layer
                lists:foreach(
                  fun(Output_Pid) ->
                    Output_Pid ! {input, {Output, self()}, Report_Pid}
                  end, Node#node.outputs)
            end,
            maps:new();
          N_Input > N_Xs ->
            % wait for more input
            Input_Values1
        end,
      loop_node(Node#node{input_values = Input_Values2});
    mark_output ->
      loop_node(Node#node{is_output = true});
    {connect, {Input_Pid, Weight}} when is_pid(Input_Pid), is_number(Weight) ->
      io:fwrite("[log] ~p connect to ~p with weight ~p~n", [Input_Pid, self(), Weight]),
      New_Inputs = maps:put(Input_Pid, Weight, Node#node.input_weights),
      loop_node(Node#node{input_weights = New_Inputs});
    X ->
      io:fwrite("[debug] ~p received ~p~n.", [self(), X]),
      loop_node(Node)
  end.

add(X, Y) ->
  X + Y.

-spec run([number()], network()) -> todo.
run(Inputs, Layers) when is_list(Layers) ->
  [Input_Layer | _] = Layers,
  Report_Pid = self(),
  F = fun(Input, Input_Pid) ->
    Input_Pid ! {input, {Input, Report_Pid}, Report_Pid}
      end,
  lists:zipwith(F, Inputs, Input_Layer),
  todo.

-spec train_all([train_data()], network()) -> todo.
train_all(Trains, Layers) when is_list(Trains), is_list(Layers) ->
  lists:foreach(fun(Train) -> train_one(Train, Layers) end, Trains),
  todo.

train_one(Train, Layers) ->
  todo.

test_xor() ->
  NN = init(2, 2, 1),
  GenTrain = fun({X, Y}) ->
    Output = if X xor Y -> 1;true -> 0 end,
    {[X, Y], [Output]}
             end,
  Trains = lists:map(GenTrain, [{1, 1}, {1, 0}, {0, 1}, {0, 0}]),
  train_all(Trains, NN).

