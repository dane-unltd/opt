package uni

type DerivSolver interface {
	Solve(obj Deriv, in *Solution, p *Params) *Result
}

type Solver interface {
	Solve(obj Function, in *Solution, p *Params) *Result
}

type DerivWrapper struct {
	S Solver
}

func (d DerivWrapper) Solve(obj Deriv, in *Solution, p *Params) *Result {
	return d.S.Solve(obj, in, p)
}
