package linprog

type Solver interface {
	Solve(m *Model)
}

func Solve(mdl *Model) {
	solver := NewPredCorr()
	solver.Solve(mdl)
}
