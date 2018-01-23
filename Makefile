PROJECT = erl_nn
PROJECT_DESCRIPTION = Erlang Neural Network
PROJECT_VERSION = 0.1.0

DEPS = erlib
dep_erlib = git https://github.com/beenotung/erlib.git v0.2.1

LOCAL_DEPS =

include erlang.mk

test:
	RELX_CONFIG=$(CURDIR)/testx.config make run
