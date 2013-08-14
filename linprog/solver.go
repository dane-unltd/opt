package linprog

func Solve(prob *Problem, p *Params, u ...Updater) *Result {
	if p == nil {
		p = NewParams()
	}
	return NewPredCorr().Solve(prob, p, u...)
}
