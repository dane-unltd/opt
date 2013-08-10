package multi

type Status int

const (
	NotTerminated (Status) = 0

	GradAbsConv = 1
	GradRelConv = 2

	ObjAbsConv = 3
	ObjRelConv = 4

	XAbsConv = 5
	XRelConv = 6

	Success = 7

	IterLimit    = -1
	TimeLimit    = -2
	FunEvalLimit = -3

	NumericalToleranceReached = -4

	Infeasible = -5

	Fail = -6
)
