package multi

import (
	"github.com/dane-unltd/linalg/mat"
)

type Solver interface {
	Solve(m *Model)
}

func Solve(f Function, x0 mat.Vec) *Model {
	mdl := NewModel(len(x0), f)
	mdl.SetX(x0, true)

	if _, ok := f.(Grad); ok {
		solver := NewLBFGS()
		mdl.Solver = solver
		solver.Solve(mdl)
	} else {
		panic("not implemented")
	}

	return mdl
}
