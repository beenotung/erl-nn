-module(ann).
-export([
  create/0
]).

%%%%%%%%
% Data %
%%%%%%%%

-type state() :: number().

-spec(state, )
-record(state, {
  %% [{pid, weight}]
  inputs :: [{pid, number()}],
  %% [pid]
  outputs :: [number()],
  bias :: number()
}).

%%%%%%%%%%%%
% Messages %
%%%%%%%%%%%%

%% check status
-record(check_status, {pid}).

%% connect two node
-record(connect_input, {pid, weight}).
-record(connect_output, {pid}).

%% pass value to input layer
-record(pass, {value}).

%% receive value from previous layer
-record(input, {pid, value}).

%%%%%%%%%%%%
% External %
%%%%%%%%%%%%

create() -> spawn(ann, loop, init()).

connect(Input_Pid, Output_Pid, Weight) ->
  Input_Pid ! #connect_output{pid = Output_Pid},
  Output_Pid ! #connect_input{pid = Input_Pid, weight = Weight}.

%%%%%%%%%%%%
% Internal %
%%%%%%%%%%%%

init() -> [#state{}].

-spec loop(State) -> State when
  State :: state().

loop(State) ->
  receive
    #check_status{pid = Pid} -> Pid ! State,
      loop(State);
    #connect_input{pid = Pid, weight = Weight} ->
      loop(State#state{inputs = [{Pid, Weight} | State#state.inputs]});
    #connect_output{pid = Pid} ->
      loop(State#state{outputs = [Pid | State#state.outputs]});
    #pass{} ->
      output =
        loop(State)
  end.

%%%%%%%%%%
% Helper %
%%%%%%%%%%

sigmoid(X) -> 1 / (1 + math:exp(-X)).

deriv_sigmoid(X) -> math:exp(-X) / (1 + math:exp(-2 * X)).

dot_prod(Xs, Ys) ->
  dot_prod(Xs, Ys, 0).

dot_prod([], [], Acc) -> Acc;
dot_prod([X | Xs], [Y | Ys], Acc) ->
  dot_prod(Xs, Ys, Acc + X * Y).
