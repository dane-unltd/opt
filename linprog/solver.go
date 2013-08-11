package linprog

func Solve(prob *Problem, p *Params, cb ...Callback) *Result {
	if p == nil {
		p = NewParams()
	}
	return NewPredCorr().Solve(prob, p, cb...)
}
