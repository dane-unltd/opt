package linprog

func Optimize(prob *Problem, p *Params, u ...Updater) *Result {
	solver := NewPredCorr()
	if p != nil {
		solver.Params = *p
	}
	return solver.Solve(prob, u...)
}
