package multi

type Status int

const (
	NotTerminated Status = iota

	GradAbsConv
	GradRelConv

	ObjAbsConv
	ObjRelConv

	XAbsConv
	XRelConv

	Success
)

const (
	IterLimit Status = -(iota + 1)
	TimeLimit
	FunEvalLimit

	NumericalToleranceReached

	Infeasible

	Fail
)
