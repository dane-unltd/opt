package uni

type Status int

const (
	NotTerminated (Status) = 0

	DerivAbsConv = 1
	DerivRelConv = 2

	ObjAbsConv = 3
	ObjRelConv = 4

	XAbsConv = 5
	XRelConv = 6

	WolfeConv = 7

	Success = 8

	IterLimit    = -1
	TimeLimit    = -2
	FunEvalLimit = -3

	NumericalToleranceReached = -4

	Infeasible = -5

	Fail = -6
)
