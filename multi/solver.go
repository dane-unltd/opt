package multi

type GradSolver interface {
	Solve(obj Grad, in *Solution, p *Params, u ...Updater) *Result
}

type ProjGradSolver interface {
	Solve(obj Grad, proj Projection,
		in *Solution, p *Params, u ...Updater) *Result
}

type Solver interface {
	Solve(obj Function, in *Solution, p *Params, u ...Updater) *Result
}

//Solve a problem choosing an appropriate solver.
//Checks the provided function for available information.
func Solve(f Function, in *Solution, p *Params, u ...Updater) *Result {
	if p == nil {
		p = NewParams()
	}

	if fg, ok := f.(Grad); ok {
		return NewLBFGS().Solve(fg, in, p, u...)
	} else {
		return NewRosenbrock().Solve(f, in, p, u...)
	}
}

//Solve a problem using a gradient based method
func SolveGrad(f Grad, in *Solution, p *Params, u ...Updater) *Result {
	if p == nil {
		p = NewParams()
	}

	return NewLBFGS().Solve(f, in, p, u...)
}

//Solve a constrained problem using a gradient and projection based method.
func SolveGradProjected(f Grad, pr Projection, in *Solution, p *Params, u ...Updater) *Result {
	if p == nil {
		p = NewParams()
	}

	return NewProjGrad().Solve(f, pr, in, p, u...)
}
