package multi

import (
	"github.com/dane-unltd/linalg/mat"
)

type Solver interface {
	Solve(m *Model)
}

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

func SolveProjected(f Function, pr Projection, x0 mat.Vec, p *Params, cb ...Callback) *Model {
	mdl := NewModel(len(x0), f)
	mdl.Proj = pr
	mdl.SetX(x0, true)

	if p != nil {
		mdl.Params = *p
	}
	if len(cb) > 0 {
		mdl.callbacks = cb
	}

	if _, ok := f.(Grad); ok {
		solver := NewProjGrad()
		mdl.Solver = solver
		solver.Solve(mdl)
	} else {
		panic("not implemented")
	}

	return mdl
}
