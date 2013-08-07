package linprog

type Status int

const (
	NotTerminated (Status) = 0

	Success = 7

	IterLimit = -1
	TimeLimit = -2

	NumericalToleranceReached = -3

	Infeasible = -4

	Fail = -5
)
