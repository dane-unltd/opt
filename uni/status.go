package uni

type Status int

const (
	NotTerminated Status = iota

	DerivAbsConv
	DerivRelConv

	ObjAbsConv
	ObjRelConv

	XAbsConv
	XRelConv

	WolfeConv

	Success
)

// Failure
const (
	IterLimit Status = -(iota + 1)
	TimeLimit
	FunEvalLimit

	NumericalToleranceReached

	Infeasible

	Fail
)
