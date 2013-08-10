package multi

import (
	"github.com/dane-unltd/linalg/mat"
)

type Solver interface {
	Solve(m *Model)
}

//Solve a problem choosing an appropriate solver.
//Checks the provided function for available information.
func Solve(f Function, x0 mat.Vec, p *Params, cb ...Callback) *Model {
	mdl := NewModel(len(x0), f)
	mdl.SetX(x0, true)

	if p != nil {
		mdl.Params = *p
	}
	if len(cb) > 0 {
		mdl.callbacks = cb
	}

	if _, ok := f.(Grad); ok {
		solver := NewLBFGS()
		mdl.Solver = solver
		solver.Solve(mdl)
	} else {
		solver := NewRosenbrock()
		mdl.Solver = solver
		solver.Solve(mdl)
	}

	return mdl
}

//Solve a problem using a gradient based method
func SolveGrad(f Grad, x0 mat.Vec, p *Params, cb ...Callback) *Model {
	mdl := NewModel(len(x0), f)
	mdl.SetX(x0, true)

	if p != nil {
		mdl.Params = *p
	}
	if len(cb) > 0 {
		mdl.callbacks = cb
	}

	solver := NewLBFGS()
	mdl.Solver = solver
	solver.Solve(mdl)

	return mdl
}

//Solve a constrained problem using a gradient and projection based method.
func SolveGradProjected(f Grad, pr Projection, x0 mat.Vec, p *Params, cb ...Callback) *Model {
	mdl := NewModel(len(x0), f)
	mdl.Proj = pr
	mdl.SetX(x0, true)

	if p != nil {
		mdl.Params = *p
	}
	if len(cb) > 0 {
		mdl.callbacks = cb
	}

	solver := NewProjGrad()
	mdl.Solver = solver
	solver.Solve(mdl)

	return mdl
}
