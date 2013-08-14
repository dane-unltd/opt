package linprog

type Status int

const (
	NotTerminated Status = iota

	Success
)

const (
	IterLimit Status = -(iota + 1)
	TimeLimit

	NumericalToleranceReached

	Infeasible

	Fail
)
